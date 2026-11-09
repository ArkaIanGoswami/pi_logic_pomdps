"""LBS, the bisection over the multiplier for (P2)"""

import time
import numpy as np
import cpi

TOL = cpi.FEAS


def best_pair(points, c, allowed=None):
    best = None
    ks = range(len(points)) if allowed is None else sorted(allowed)
    feas = [k for k in ks if points[k][0] >= c - TOL]
    for j in feas:
        if best is None or points[j][1] > best[0]:
            best = (points[j][1], j, j, 1.0)
    low = [k for k in ks if points[k][0] < c - TOL]
    for i in low:
        pi, ri = points[i]
        for j in feas:
            pj, rj = points[j]
            th = (c - pi) / (pj - pi)
            v = th * rj + (1 - th) * ri
            if v > best[0]:
                best = (v, i, j, th)
    return best


def upper_hull(points):
    order = sorted(range(len(points)), key=lambda k: (points[k][0], -points[k][1]))
    hull = []
    for k in order:
        while len(hull) >= 2:
            (x1, y1), (x2, y2), (x3, y3) = points[hull[-2]], points[hull[-1]], points[k]
            if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) >= -1e-15:
                hull.pop()
            else:
                break
        hull.append(k)
    best_phi = max(range(len(points)), key=lambda k: (points[k][0], points[k][1]))
    if best_phi not in hull:
        hull.append(best_phi)
    return hull


class Incumbent:
    def __init__(self, c):
        self.c, self.points, self.tags, self.pols = c, [], [], {}
        self.sweeps, self.curve, self.best = 0, [], None

    def add(self, J, pol, tag, sweep=True):
        k = len(self.points)
        self.points.append((float(J['phi']), float(J['r'])))
        self.tags.append(tag)
        hull = set(upper_hull(self.points))
        if k in hull:
            self.pols[k] = pol.copy()
        for gone in [i for i in self.pols if i not in hull]:
            del self.pols[gone]
        self.best = best_pair(self.points, self.c, allowed=self.pols)
        if sweep:
            self.sweeps += 1
        v = self.best[0] if self.best is not None else np.nan
        self.curve.append((self.sweeps, v))
        return k

    def mixture(self, task):
        v, i, j, th = self.best
        return cpi.mixture(task, self.pols[j], self.pols[i], th)


def make_oracle(task, model, warm, start, cfg, inc, log):
    alpha = cfg.get('inner_alpha', 1.0)
    inner = dict(cfg, alpha=alpha, lazy=alpha == 1.0)

    def ascent(pol, lam, tag):
        J0 = cpi.evaluate(task, model, pol)
        J = dict(phi=J0['J_phi'], r=J0['J_r'])
        sub = []
        hook = lambda p, Jc, lg: (inc.add(Jc, p, tag), False)[1]
        J = cpi.converge(task, model, pol, 'lagrange', J, inner, sub, tag, lam=lam, hook=hook)
        return pol, J, len(sub)

    def d(lam):
        t0 = time.time()
        cands = [ascent(p.copy(), lam, f'{tag}@{lam:.6g}') for p, tag in ((warm, 'warm'), (start, 'init'))]
        pol, J, _ = max(cands, key=lambda c: c[1]['r'] + lam * c[1]['phi'])
        src = 'ascent'
        if inc.pols:
            k = max(inc.pols, key=lambda i: inc.points[i][1] + lam * inc.points[i][0])
            phi_k, r_k = inc.points[k]
            L_now = J['r'] + lam * J['phi']
            if r_k + lam * phi_k > L_now + 1e-12 * (1 + abs(L_now)):
                pol, J, src = inc.pols[k].copy(), dict(phi=phi_k, r=r_k), 'hull'
        log.append(dict(lam=float(lam), J_phi=J['phi'], J_r=J['r'], L=J['r'] + lam * J['phi'], source=src,
                        sweeps=sum(c[2] for c in cands), secs=round(time.time() - t0, 3),
                        sweeps_total=inc.sweeps))
        return pol, J
    return d


def reward_span(task):
    span = task['env']['info'].get('reward_span')
    return float(span) if span is not None else cpi.reward_span(task)


def epsilon(task, lam, L_value):
    f = task['env'].get('max_lagrangian')
    return None if f is None else float(f(lam, task['memory_size']) - L_value)


def warm_start(task, model, start, cfg, inc):
    pol = start.copy()
    J0 = cpi.evaluate(task, model, pol)
    J = dict(phi=J0['J_phi'], r=J0['J_r'])
    inc.add(J, pol, 'start', sweep=False)
    lg = []
    rng = np.random.default_rng(cfg.get('seed', 0)) if cfg.get('ties') == 'random' else None
    cpi.converge(dict(task, objective='P1'), model, pol, 'sat', J, dict(cfg, lazy=cfg.get('alpha', 0.9) == 1.0),
                 lg, 'P1', rng=rng, hook=lambda p, Jc, l: (inc.add(Jc, p, 'P1'), False)[1])
    return pol, len(lg)


def bisection(task, model, c, start, cfg, steps=14, rule='mid', polish=True):
    t_start = time.time()
    inc, log = Incumbent(c), []
    warm, p1_sweeps = warm_start(task, model, start, cfg, inc)
    Jw = cpi.evaluate(task, model, warm)
    kappa = Jw['J_phi'] - c
    out = dict(method='bisection', rule=rule, c=c, kappa=kappa, R=reward_span(task), p1_sweeps=p1_sweeps,
               warm_J_phi=Jw['J_phi'], warm_J_r=Jw['J_r'])
    if kappa <= 0:
        return dict(out, status='warm start infeasible', incumbent=inc, log=log, curve=list(inc.curve))
    d = make_oracle(task, model, warm, start, cfg, inc, log)
    d_lo, J_lo = d(0.0)
    lam_lo = 0.0
    if J_lo['phi'] >= c - TOL:
        out.update(status='slack', theta=1.0, lam_lo=0.0, lam_hi=0.0, width=0.0, certificate=0.0,
                   bracket_J_phi=J_lo['phi'], bracket_J_r=J_lo['r'], J_hi=J_lo, J_lo=J_lo,
                   eps_lo=epsilon(task, 0.0, J_lo['r']), eps_hi=None)
        return _finish(task, model, c, out, inc, log, polish, cfg, t_start)
    lam_hi = out['R'] / kappa
    d_hi, J_hi = d(lam_hi)
    bracket = [dict(step=0, lam_lo=lam_lo, lam_hi=lam_hi, J_phi_lo=J_lo['phi'], J_phi_hi=J_hi['phi'],
                    sweeps_total=inc.sweeps)]
    if J_hi['phi'] < c - TOL:
        return dict(out, status='upper multiplier infeasible', incumbent=inc, log=log, bracket=bracket,
                    curve=list(inc.curve))
    for k in range(steps):
        if rule == 'mid':
            mid = 0.5 * (lam_lo + lam_hi)
        else:
            mid = np.sqrt(lam_lo * lam_hi) if lam_lo > 0 else lam_hi / 2 ** 10
        p, Jm = d(mid)
        if Jm['phi'] >= c - TOL:
            lam_hi, d_hi, J_hi = mid, p, Jm
        else:
            lam_lo, d_lo, J_lo = mid, p, Jm
        bracket.append(dict(step=k + 1, lam_lo=lam_lo, lam_hi=lam_hi, J_phi_lo=J_lo['phi'], J_phi_hi=J_hi['phi'],
                            sweeps_total=inc.sweeps))
    theta = (c - J_lo['phi']) / (J_hi['phi'] - J_lo['phi'])
    width = lam_hi - lam_lo
    out.update(status='mixed', theta=theta, lam_lo=lam_lo, lam_hi=lam_hi, width=width,
               certificate=width * theta * (1 - theta) * (J_hi['phi'] - J_lo['phi']),
               bracket_J_phi=theta * J_hi['phi'] + (1 - theta) * J_lo['phi'],
               bracket_J_r=theta * J_hi['r'] + (1 - theta) * J_lo['r'],
               J_hi=J_hi, J_lo=J_lo, bracket=bracket,
               eps_lo=epsilon(task, lam_lo, J_lo['r'] + lam_lo * J_lo['phi']),
               eps_hi=epsilon(task, lam_hi, J_hi['r'] + lam_hi * J_hi['phi']))
    return _finish(task, model, c, out, inc, log, polish, cfg, t_start)


def _finish(task, model, c, out, inc, log, polish, cfg, t_start):
    mix, mix_task = inc.mixture(task)
    J = cpi.evaluate(mix_task, model, mix)
    v, i, j, th = inc.best
    out.update(J_phi=J['J_phi'], J_r=J['J_r'], pair_value=v, pair_theta=th,
               pair_tags=(inc.tags[i], inc.tags[j]), pair_points=(inc.points[i], inc.points[j]),
               points=len(inc.points), sweeps=inc.sweeps, secs=time.time() - t_start,
               curve_check=abs(J['J_r'] - v))
    curve = list(inc.curve)
    pol_final, task_final = mix, mix_task
    if polish:
        plog = []
        mix = mix.copy()
        Jp = cpi.converge(mix_task, model, mix, 'joint', dict(phi=J['J_phi'], r=J['J_r']),
                          dict(cfg, alpha=cfg.get('alpha', 0.9), lazy=False), plog, 'polish', c=c)
        n0 = inc.sweeps
        for k, rec in enumerate(plog):
            curve.append((n0 + k + 1, max(v, rec['J_r']) if rec['J_phi'] >= c - 1e-9 else v))
        out.update(polish_J_phi=Jp['phi'], polish_J_r=Jp['r'], polish_sweeps=len(plog),
                   secs_with_polish=time.time() - t_start)
        pol_final = mix
    out.update(curve=curve, incumbent=inc, log=log, policy=pol_final, policy_task=task_final)
    return out


def exponentiated_gradient(task, model, c, start, cfg, iters=16, eta=None, polish=False):
    t_start = time.time()
    inc, log = Incumbent(c), []
    warm, p1_sweeps = warm_start(task, model, start, cfg, inc)
    Jw = cpi.evaluate(task, model, warm)
    kappa = Jw['J_phi'] - c
    out = dict(method='eg', c=c, kappa=kappa, R=reward_span(task), p1_sweeps=p1_sweeps, iters=iters,
               warm_J_phi=Jw['J_phi'], warm_J_r=Jw['J_r'])
    if kappa <= 0:
        return dict(out, status='warm start infeasible', incumbent=inc, log=log, curve=list(inc.curve))
    B = out['R'] / kappa
    eta = float(np.sqrt(8 * np.log(2) / iters) / 2) if eta is None else float(eta)
    d = make_oracle(task, model, warm, start, cfg, inc, log)
    l, phis, rs, lams, avg_curve = 0.0, [], [], [], []
    for k in range(iters):
        lam = B / (1 + np.exp(-l))
        _, J = d(lam)
        lams.append(lam); phis.append(J['phi']); rs.append(J['r'])
        l += eta * (c - J['phi'])
        mphi, mr = float(np.mean(phis)), float(np.mean(rs))
        avg_curve.append((inc.sweeps, mr if mphi >= c - TOL else np.nan, mphi))
    mphi, mr = float(np.mean(phis)), float(np.mean(rs))
    v, i, j, th = inc.best
    mix, mix_task = inc.mixture(task)
    Jm = cpi.evaluate(mix_task, model, mix)
    out.update(status='done', B=B, eta=eta, lams=lams, iter_J_phi=phis, iter_J_r=rs,
               avg_J_phi=mphi, avg_J_r=mr, avg_violation=max(0.0, c - mphi),
               avg_policies=len(set(zip(np.round(phis, 12), np.round(rs, 12)))),
               avg_bits=int(np.ceil(np.log2(iters))) if iters > 1 else 0,
               J_phi=Jm['J_phi'], J_r=Jm['J_r'], pair_value=v, pair_theta=th,
               pair_tags=(inc.tags[i], inc.tags[j]), points=len(inc.points), sweeps=inc.sweeps,
               secs=time.time() - t_start, curve=list(inc.curve), avg_curve=avg_curve,
               incumbent=inc, log=log, policy=mix, policy_task=mix_task)
    return out
