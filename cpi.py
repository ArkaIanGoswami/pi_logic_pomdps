"""Conservative policy iteration (CPI) for (P1) and (P2) on the exact product model."""

import ast
import time
import numpy as np
from scipy import sparse
from policies import dirichlet_row

MASS = 1e-13
TIE = 1e-12
FEAS = 1e-10


def aux(model, Z):
    cache = model.setdefault('_aux', {})
    if Z not in cache:
        n, R = model['n'], len(model['rows'])
        G = sparse.csr_matrix((np.ones(n), (model['row_of_node'], np.arange(n))), shape=(R, n))
        legal = np.repeat(model['legal'][:, :, None], Z, axis=2).reshape(R, -1)
        cache[Z] = dict(G=G, legal=legal)
    return cache[Z]


def start_occupancy(task, model):
    return model['initial'][:, None] * np.asarray(task['initial_memory'], float)[None, :]


def stacked(model):
    if '_stack' not in model:
        Pall = sparse.vstack(model['P']).tocsr()
        model['_stack'] = (Pall, Pall.T.tocsr())
    return model['_stack']


def forward(model, pol_t, mu):
    act = np.einsum('nz,nzaw->naw', mu, pol_t[model['row_of_node']])
    _, PT = stacked(model)
    return PT @ np.ascontiguousarray(act.transpose(1, 0, 2)).reshape(-1, mu.shape[1])


def occupancies(task, model, pol):
    mus = [start_occupancy(task, model)]
    for t in range(task['horizon'] - 1):
        mus.append(forward(model, pol[t], mus[-1]))
    return mus


def qvalues(model, V, reward_weight=0.0):
    Pall, _ = stacked(model)
    Q = (Pall @ np.ascontiguousarray(V)).reshape(model['A'], model['n'], -1).transpose(1, 0, 2)
    if reward_weight:
        Q += reward_weight * model['R'][:, :, None]
    return Q


def node_values(model, pol_t, Q):
    return np.einsum('nzaw,naw->nz', pol_t[model['row_of_node']], Q)


def row_average(G, mu, Q):
    mass = G @ mu
    Qf = Q.reshape(Q.shape[0], -1)
    avg = np.stack([G @ (mu[:, z, None] * Qf) for z in range(mu.shape[1])], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return mass, avg / mass[:, :, None]


def evaluate(task, model, pol):
    mu, J_r = start_occupancy(task, model), 0.0
    for t in range(task['horizon']):
        act = np.einsum('nz,nzaw->na', mu, pol[t][model['row_of_node']])
        J_r += task['gamma'] ** t * float((act * model['R']).sum())
        mu = forward(model, pol[t], mu)
    return dict(J_phi=float(mu[model['accepting']].sum()), J_reject=float(mu[model['rejecting']].sum()), J_r=J_r)


def initial_policy(task, model, kind='dirichlet', seed=0):
    H, A, Z = task['horizon'], model['A'], task['memory_size']
    rows, legal = model['rows'], model['legal']
    pol = np.zeros((H, len(rows), Z, A, Z))
    if kind == 'dirichlet':
        for t in range(H):
            for r, (y, q) in enumerate(rows):
                for z in range(Z):
                    p = dirichlet_row(seed, (t, y, q, z), A, Z)
                    p[~legal[r]] = 0.0
                    pol[t, r, z] = p / p.sum()
        return pol
    act = legal / legal.sum(axis=1, keepdims=True)
    if kind == 'uniform':
        pol[:] = act[None, :, None, :, None] / Z
    elif kind == 'copy':
        for z in range(Z):
            pol[:, :, z, :, z] = act[None]
    elif kind == 'copycoin':
        for z in range(Z):
            pol[1:, :, z, :, z] = act[None]
        pol[0] = act[:, None, :, None] / Z
    else:
        raise ValueError(kind)
    return pol


def transfer(pol, rows_from, rows_to, fallback, key=lambda row: row):
    index = {key(r): i for i, r in enumerate(rows_from)}
    out = fallback.copy()
    for j, row in enumerate(rows_to):
        i = index.get(key(row))
        if i is not None:
            out[:, j] = pol[:, i]
    return out


def save_policy(path, model, pol, task, **extra):
    np.savez_compressed(path, pol=pol.astype(np.float32), rows=np.array([repr(r) for r in model['rows']]),
                        initial_memory=np.asarray(task['initial_memory'], float), **extra)


def load_policy(path):
    d = np.load(path, allow_pickle=False)
    return dict(pol=d['pol'].astype(float), rows=[ast.literal_eval(r) for r in d['rows']],
                initial_memory=d['initial_memory'])


def as_function(saved):
    index = {row: i for i, row in enumerate(saved['rows'])}
    pol = saved['pol']

    def pi(t, y, q, z):
        i = index.get((y, q))
        return np.ones(pol.shape[3:]) if i is None else pol[t, i, z]
    return pi


def greedy(primary, legal, secondary=None, rng=None):
    s = np.where(legal, primary, -np.inf)
    m = s.max(axis=1, keepdims=True)
    top = s >= m - TIE * (1 + np.abs(m))
    cand = top
    if secondary is not None:
        s2 = np.where(top, secondary, -np.inf)
        m2 = s2.max(axis=1, keepdims=True)
        cand = top & (s2 >= m2 - TIE * (1 + np.abs(m2)))
    pick = rng.random(cand.shape) * cand if rng is not None else cand
    out = np.zeros_like(primary)
    out[np.arange(len(out)), np.argmax(pick, axis=1)] = 1.0
    return out, top


def joint_rule(old, w, qr, qp, legal, J, c, alpha):
    base = float(w @ (old * qp).sum(1))
    g, _ = greedy(qr, legal, qp)
    S = J + alpha * (float(w @ (g * qp).sum(1)) - base)
    if S >= c - FEAS:
        return g, 0.0, 0, 0.0
    target = min(c, J) if J >= c - FEAS else c
    k = len(qr)
    rows = np.arange(k)
    choice = g.argmax(1)
    cur, lam, active = choice.copy(), np.zeros(k), np.ones(k, bool)
    ev = [[], [], [], []]
    for _ in range(qr.shape[1]):
        if not active.any():
            break
        cp, cr = qp[rows, cur], qr[rows, cur]
        dp = qp - cp[:, None]
        cand = legal & active[:, None] & (dp > TIE * (1 + np.abs(cp))[:, None])
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(cand, np.maximum((cr[:, None] - qr) / dp, lam[:, None]), np.inf)
        nl = ratio.min(1)
        active &= np.isfinite(nl)
        tie = cand & (ratio <= nl[:, None] + TIE * (1 + np.abs(nl[:, None])))
        nxt = np.where(tie, qp, -np.inf).argmax(1)
        for lst, v in zip(ev, (nl[active], rows[active], cur[active], nxt[active])):
            lst.append(v)
        cur, lam = np.where(active, nxt, cur), np.where(active, nl, lam)
    L, R_, F, T = (np.concatenate(e) if e else np.zeros(0) for e in ev)
    lam_t, mixed, nu = np.inf, 0, None
    for e in np.argsort(L, kind='stable'):
        r, f, to = int(R_[e]), int(F[e]), int(T[e])
        dS = alpha * w[r] * (qp[r, to] - qp[r, f])
        if dS > 0 and S + dS >= c - FEAS:
            theta = min(1.0, max(0.0, (target - S) / dS))
            nu = np.zeros_like(qr)
            nu[rows, choice] = 1.0
            nu[r, f], nu[r, to] = 1.0 - theta, theta
            lam_t, mixed = float(L[e]), 1
            break
        S += dS
        choice[r] = to
    if nu is None:
        nu = np.zeros_like(qr)
        nu[rows, choice] = 1.0
        return nu, np.inf, 0, 0.0
    s = np.where(legal, qr + lam_t * qp, -np.inf)
    gain = (np.where(nu > 0, s, 0.0) * nu).sum(1)
    return nu, lam_t, mixed, float(w @ (s.max(1) - gain)) / (1.0 + lam_t)


def naive_rule(old, w, qr, qp, legal, J, c, alpha):
    g, _ = greedy(qr, legal, qp)
    drop = float(w @ ((g - old) * qp).sum(1))
    beta = alpha if drop >= 0 or J + alpha * drop >= c - FEAS else min(alpha, max(0.0, (J - c) / -drop))
    return (1 - beta) * old + beta * g, beta


def stage_rule(mode, old, w, qr, qp, legal, J_phi, c, alpha, lazy, rng, lam):
    info = {}
    if mode == 'joint':
        nu, lam_t, mixed, gap = joint_rule(old, w, qr, qp, legal, J_phi, c, alpha)
        info = dict(lam=lam_t, mixed=mixed, gap=gap)
        return (1 - alpha) * old + alpha * nu, info
    if mode == 'naive':
        new, beta = naive_rule(old, w, qr, qp, legal, J_phi, c, alpha)
        return new, dict(lam=0.0, mixed=0, gap=0.0, short=int(beta < alpha))
    if mode == 'sat':
        g, top = greedy(qp, legal, qr, rng)
    elif mode == 'reward':
        g, top = greedy(qr, legal, None, rng)
    elif mode == 'lagrange':
        g, top = greedy(qr + lam * qp, legal, qp, rng)
    else:
        raise ValueError(mode)
    new = (1 - alpha) * old + alpha * g
    if lazy:
        keep = ~((old > 0) & ~top).any(axis=1)
        new[keep] = old[keep]
    return new, info


def sweep(task, model, pol, mode, J, c=None, alpha=0.9, lazy=False, rng=None, lam=0.0):
    H, gamma, Z, A = task['horizon'], task['gamma'], pol.shape[2], model['A']
    ax = aux(model, Z)
    mus = occupancies(task, model, pol)
    Vp = np.repeat(model['accepting'][:, None].astype(float), Z, axis=1)
    Vr = np.zeros_like(Vp)
    run_phi, run_r = J['phi'], J['r']
    st = dict(changed=0, live=0, occupied=0, min_phi=run_phi, rejected=0, lam_pos=0, lam_inf=0, lam_max=0.0,
              mixed=0, gap=0.0, stages=0, short=0)
    for t in range(H - 1, -1, -1):
        Qp = qvalues(model, Vp)
        Qr = qvalues(model, Vr, gamma ** t)
        mass, Qp_bar = row_average(ax['G'], mus[t], Qp)
        _, Qr_bar = row_average(ax['G'], mus[t], Qr)
        live = np.argwhere(mass > MASS)
        if len(live):
            rr, zz = live[:, 0], live[:, 1]
            old = pol[t][rr, zz].reshape(len(live), -1)
            w, qp, qr = mass[rr, zz], Qp_bar[rr, zz], Qr_bar[rr, zz]
            new, info = stage_rule(mode, old, w, qr, qp, ax['legal'][rr], run_phi, c, alpha, lazy, rng, lam)
            d_phi = float(w @ ((new - old) * qp).sum(1))
            d_r = float(w @ ((new - old) * qr).sum(1))
            if mode in ('joint', 'naive') and run_phi + d_phi < c - 1e-9:
                new, d_phi, d_r = old, 0.0, 0.0
                st['rejected'] += 1
            pol[t][rr, zz] = new.reshape(-1, A, Z)
            run_phi, run_r = run_phi + d_phi, run_r + d_r
            st['min_phi'] = min(st['min_phi'], run_phi)
            st['changed'] += int((np.abs(new - old).max(axis=1) > 0).sum())
            st['live'] += len(live)
            st['occupied'] += int((w > 1e-6).sum())
            if info:
                st['stages'] += 1
                st['lam_pos'] += int(info['lam'] > 0)
                st['lam_inf'] += int(np.isinf(info['lam']))
                if np.isfinite(info['lam']):
                    st['lam_max'] = max(st['lam_max'], info['lam'])
                st['mixed'] += info['mixed']
                st['gap'] = max(st['gap'], abs(info['gap']))
                st['short'] += info.get('short', 0)
        Vp, Vr = node_values(model, pol[t], Qp), node_values(model, pol[t], Qr)
    mu0 = mus[0]
    J_new = dict(phi=float((mu0 * Vp).sum()), r=float((mu0 * Vr).sum()))
    st.update(J_phi=J_new['phi'], J_r=J_new['r'],
              identity_err=max(abs(J_new['phi'] - run_phi), abs(J_new['r'] - run_r) / max(1.0, abs(J_new['r']))))
    return J_new, st


def converge(task, model, pol, mode, J, cfg, log, phase, c=None, lam=0.0, rng=None, on_sweep=None, hook=None):
    lazy = cfg.get('lazy', False)
    score = {'sat': lambda j: j['phi'], 'reward': lambda j: j['r'], 'joint': lambda j: j['r'], 'naive': lambda j: j['r'],
             'lagrange': lambda j: j['r'] + lam * j['phi']}[mode]
    stall = 0
    for k in range(cfg.get('max_sweeps', 60)):
        t0 = time.time()
        J_new, st = sweep(task, model, pol, mode, J, c, cfg.get('alpha', 0.9), lazy, rng, lam)
        gain = score(J_new) - score(J)
        J = J_new
        st.update(phase=phase, sweep=k + 1, secs=round(time.time() - t0, 3))
        if on_sweep is not None:
            st.update(on_sweep(pol))
        log.append(st)
        if hook is not None and hook(pol, J, log):
            break
        if st['changed'] == 0:
            break
        stall = stall + 1 if gain <= cfg.get('tol', 1e-9) * max(1.0, abs(score(J))) else 0
        if stall >= 2:
            break
    return J


def solve(task, model, cfg, levels=(), pol=None, rng=None, on_sweep=None):
    if pol is None:
        pol = initial_policy(task, model, cfg.get('start', 'dirichlet'), task['seed'])
    rng = rng if cfg.get('ties') == 'random' else None
    J0 = evaluate(task, model, pol)
    J = dict(phi=J0['J_phi'], r=J0['J_r'])
    log = [dict(phase='init', sweep=0, J_phi=J['phi'], J_r=J['r'], **(on_sweep(pol) if on_sweep else {}))]
    if task['objective'] == 'P0':
        J = converge(task, model, pol, 'reward', J, cfg, log, 'P0', rng=rng, on_sweep=on_sweep)
        return {'P0': dict(pol=pol, J=J, log=log, feasible_start=True)}
    if task['objective'] == 'P1':
        J = converge(task, model, pol, 'sat', J, cfg, log, 'P1', rng=rng, on_sweep=on_sweep)
        return {'P1': dict(pol=pol, J=J, log=log, feasible_start=True)}
    branches = {}

    def hook(p, Jc, lg):
        for name, c in levels:
            if name not in branches and Jc['phi'] >= c - 1e-12:
                branches[name] = (p.copy(), dict(Jc), list(lg), len(lg))
        return len(branches) == len(levels)

    if cfg.get('p2_from') == 'converged':
        J = converge(task, model, pol, 'sat', J, cfg, log, 'P1', rng=rng, on_sweep=on_sweep)
        hook(pol, J, log)
    elif not hook(pol, J, log):
        J = converge(task, model, pol, 'sat', J, cfg, log, 'P1', rng=rng, on_sweep=on_sweep, hook=hook)
    out, reuse = {}, {}
    for name, c in sorted(levels, key=lambda lc: -lc[1]):
        if name not in branches:
            out[name] = dict(pol=pol.copy(), J=dict(J), log=list(log), feasible_start=False, c=c)
            continue
        p, Jb, lg, key = branches[name]
        if key in reuse:
            r = reuse[key]
            out[name] = dict(r, pol=r['pol'].copy(), log=list(r['log']), c=c, reused=True)
            continue
        Jc = converge(task, model, p, cfg.get('p2_step', 'joint'), Jb, cfg, lg, 'P2', c=c, on_sweep=on_sweep)
        out[name] = dict(pol=p, J=Jc, log=lg, feasible_start=True, c=c)
        if all(l.get('lam_pos', 0) == 0 and l.get('short', 0) == 0 and l.get('rejected', 0) == 0
               for l in lg if l['phase'] == 'P2'):
            reuse[key] = out[name]
    return {name: out[name] for name, _ in levels}


def reward_span(task):
    lo, hi = task['env']['reward_range']
    return (hi - lo) * sum(task['gamma'] ** t for t in range(task['horizon']))


def block_ascent(task, model, pol, lam, cfg, log, tag):
    J0 = evaluate(task, model, pol)
    J = dict(phi=J0['J_phi'], r=J0['J_r'])
    sub = []
    a = cfg.get('inner_alpha', 1.0)
    J = converge(task, model, pol, 'lagrange', J, dict(cfg, alpha=a, lazy=a == 1.0), sub, tag, lam=lam)
    log.append(dict(tag=tag, lam=lam, sweeps=len(sub), J_phi=J['phi'], J_r=J['r'], L=J['r'] + lam * J['phi']))
    return pol, J


def lagrangian(task, model, c, warm, start, cfg, steps=14):
    log = []
    Jw = evaluate(task, model, warm)
    kappa = Jw['J_phi'] - c
    if kappa <= 0:
        return dict(status='warm start infeasible', log=log, kappa=kappa)

    def d(lam):
        cands = [block_ascent(task, model, p.copy(), lam, cfg, log, tag) for p, tag in ((warm, 'warm'), (start, 'init'))]
        return max(cands, key=lambda pj: pj[1]['r'] + lam * pj[1]['phi'])

    lam_hi = reward_span(task) / kappa
    d_lo, J_lo = d(0.0)
    if J_lo['phi'] >= c:
        return dict(status='slack', theta=1.0, J_phi=J_lo['phi'], J_r=J_lo['r'], lam_lo=0.0, lam_hi=0.0,
                    log=log, kappa=kappa, pol_hi=d_lo, pol_lo=d_lo)
    d_hi, J_hi = d(lam_hi)
    bracket = [dict(step=0, lam_lo=0.0, lam_hi=lam_hi, width=lam_hi, J_phi_hi=J_hi['phi'], J_phi_lo=J_lo['phi'])]
    lam_lo = 0.0
    for k in range(steps):
        mid = np.sqrt(lam_lo * lam_hi) if lam_lo > 0 else lam_hi / 2 ** 10
        p, Jm = d(mid)
        if Jm['phi'] >= c:
            lam_hi, d_hi, J_hi = mid, p, Jm
        else:
            lam_lo, d_lo, J_lo = mid, p, Jm
        bracket.append(dict(step=k + 1, lam_lo=lam_lo, lam_hi=lam_hi, width=lam_hi - lam_lo,
                            J_phi_hi=J_hi['phi'], J_phi_lo=J_lo['phi']))
    theta = (c - J_lo['phi']) / (J_hi['phi'] - J_lo['phi'])
    mix, mix_task = mixture(task, d_hi, d_lo, theta)
    Jm = evaluate(mix_task, model, mix)
    return dict(status='mixed', theta=theta, J_phi=Jm['J_phi'], J_r=Jm['J_r'],
                J_r_formula=theta * J_hi['r'] + (1 - theta) * J_lo['r'], lam_lo=lam_lo, lam_hi=lam_hi,
                width=lam_hi - lam_lo, kappa=kappa, bracket=bracket, log=log, pol_hi=d_hi, pol_lo=d_lo,
                J_hi=J_hi, J_lo=J_lo, mixture=mix, mixture_task=mix_task)


def mixture(task, d_hi, d_lo, theta):
    H, R, Z, A, _ = d_hi.shape
    pol = np.zeros((H, R, 2 * Z, A, 2 * Z))
    pol[:, :, :Z, :, :Z] = d_hi
    pol[:, :, Z:, :, Z:] = d_lo
    z0 = np.asarray(task['initial_memory'], float)
    mix_task = dict(task, memory_size=2 * Z, initial_memory=tuple(np.concatenate([theta * z0, (1 - theta) * z0])))
    return pol, mix_task


def memory_information(task, model, pol, feature):
    mu, best, when = start_occupancy(task, model), 0.0, 0
    labels = np.array([feature(s) for s, _, _ in model['nodes']])
    values = sorted(set(labels.tolist()))
    for t in range(task['horizon']):
        mu = forward(model, pol[t], mu)
        joint = np.array([mu[labels == v].sum(axis=0) for v in values])
        joint = joint / joint.sum()
        px, pz = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            mi = float(np.nansum(joint * np.log2(joint / (px * pz))))
        if mi > best:
            best, when = mi, t + 1
    return best, when
