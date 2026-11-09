"""evaluation of belief-space policies for CGCP, the reference for (P2) without a memory budget."""

import time
from collections import defaultdict
import numpy as np
from scipy.optimize import linprog
import lbs

FEAS = 1e-9


class BeliefModel:
    def __init__(self, task, model):
        self.H, self.gamma = task['horizon'], task['gamma']
        rof = model['row_of_node']
        self.nR, self.A = len(model['rows']), model['A']
        self.nodes_of = [np.flatnonzero(rof == r) for r in range(self.nR)]
        self.legal = model['legal']
        self.Rw = [model['R'][ix] for ix in self.nodes_of]
        self.acc = [model['accepting'][ix].astype(float) for ix in self.nodes_of]
        P = [p.tocsr() for p in model['P']]
        self.T = [[None] * self.A for _ in range(self.nR)]
        for r, ix in enumerate(self.nodes_of):
            for a in range(self.A):
                if not self.legal[r, a]:
                    continue
                sub = P[a][ix]
                nxt = []
                for r2 in np.unique(rof[np.unique(sub.indices)]):
                    B = sub[:, self.nodes_of[r2]].toarray()
                    if B.any():
                        nxt.append((int(r2), B))
                self.T[r][a] = nxt
        init = model['initial']
        self.roots = []
        for r, ix in enumerate(self.nodes_of):
            m = init[ix].sum()
            if m > 0:
                self.roots.append((r, float(m), init[ix] / m))

    def w(self, t):
        return self.gamma ** t

    def default(self, lam):
        V = [self.acc[r] * lam for r in range(self.nR)]
        out = [None] * self.H
        for t in range(self.H - 1, -1, -1):
            Vt, acts = [None] * self.nR, np.zeros(self.nR, int)
            for r in range(self.nR):
                best, ba = None, 0
                for a in np.flatnonzero(self.legal[r]):
                    q = self.w(t) * self.Rw[r][:, a]
                    for r2, B in self.T[r][a]:
                        q = q + B @ V[r2]
                    if best is None or q.mean() > best.mean():
                        best, ba = q, a
                Vt[r], acts[r] = best, ba
            out[t] = (Vt, acts)
            V = Vt
        return out

    def seed_alphas(self, task, model, pol, lam):
        import cpi
        V = np.repeat((model['accepting'].astype(float) * lam)[:, None], pol.shape[2], axis=1)
        out = [None] * self.H
        for t in range(self.H - 1, -1, -1):
            V = cpi.node_values(model, pol[t], cpi.qvalues(model, V, self.gamma ** t))
            out[t] = [V[ix].T.copy() for ix in self.nodes_of]
        return out

    def solve(self, lam, rng, seeds=(), rounds=8, rollouts=300, cap=32, eps=(1.0, 0.1), tol=1e-9, patience=2):
        H = self.H
        dflt = self.default(lam)
        fixed = [{r: dflt[t][0][r][None, :] for r in range(self.nR)} for t in range(H)]
        for sd in seeds:
            for t in range(H):
                for r in range(self.nR):
                    fixed[t][r] = np.vstack([fixed[t][r], sd[t][r]])
        Gamma = [dict(f) for f in fixed]
        Gamma.append({r: (self.acc[r] * lam)[None, :] for r in range(self.nR)})
        points = [defaultdict(dict) for _ in range(H)]
        best_val, stall = -np.inf, 0
        for k in range(rounds):
            e = eps[0] + (eps[1] - eps[0]) * k / max(1, rounds - 1)
            self._collect(Gamma, points, rng, rollouts, e, cap)
            for t in range(H - 1, -1, -1):
                for r, pts in points[t].items():
                    G = np.vstack([fixed[t][r], self._backup(Gamma, t, r, np.array(list(pts.values())))])
                    Gamma[t][r] = np.unique(np.round(G, 12), axis=0)
            val = self.evaluate(Gamma, lam)['L']
            stall = stall + 1 if val <= best_val + tol * (1 + abs(best_val)) else 0
            best_val = max(best_val, val)
            if stall >= patience:
                break
        return Gamma

    def _backup(self, Gamma, t, r, Pm):
        m, k = Pm.shape
        best_v, best_al = np.full(m, -np.inf), np.zeros((m, k))
        for a in np.flatnonzero(self.legal[r]):
            al = np.repeat((self.w(t) * self.Rw[r][:, a])[None, :], m, axis=0)
            for r2, B in self.T[r][a]:
                G2 = Gamma[t + 1][r2]
                j = np.argmax((Pm @ B) @ G2.T, axis=1)
                al = al + (B @ G2[j].T).T
            v = (al * Pm).sum(1)
            up = v > best_v
            best_v[up], best_al[up] = v[up], al[up]
        return best_al

    def lookahead(self, Gamma, t, r, b):
        best, ba = -np.inf, None
        for a in np.flatnonzero(self.legal[r]):
            v = self.w(t) * float(b @ self.Rw[r][:, a])
            for r2, B in self.T[r][a]:
                v += float(np.max(Gamma[t + 1][r2] @ (b @ B)))
            if ba is None or v > best + 1e-12 * (1 + abs(best)):
                best, ba = v, int(a)
        return ba

    greedy = lookahead

    def _collect(self, Gamma, points, rng, rollouts, eps, cap):
        def add(t, r, b):
            d = points[t][r]
            key = tuple(np.round(b, 9))
            if key not in d and len(d) < cap:
                d[key] = b
        for t, r, b, _ in self.tree(Gamma):
            if t < self.H:
                add(t, r, b)
        roots_p = np.array([m for _, m, _ in self.roots])
        for _ in range(rollouts):
            r, _, b = self.roots[rng.choice(len(self.roots), p=roots_p / roots_p.sum())]
            for t in range(self.H):
                add(t, r, b)
                legal = np.flatnonzero(self.legal[r])
                a = self.lookahead(Gamma, t, r, b) if rng.random() > eps else int(rng.choice(legal))
                nxt = [(r2, b @ B) for r2, B in self.T[r][a]]
                ms = np.array([v.sum() for _, v in nxt])
                if ms.sum() <= 0:
                    break
                k = rng.choice(len(nxt), p=ms / ms.sum())
                r, b = nxt[k][0], nxt[k][1] / ms[k]

    def tree(self, Gamma):
        layer = {}
        for r, m, b in self.roots:
            layer[(r, tuple(np.round(b, 12)))] = [r, m, b]
        for t in range(self.H):
            new = {}
            for r, m, b in layer.values():
                yield t, r, b, m
                a = self.greedy(Gamma, t, r, b)
                for r2, B in self.T[r][a]:
                    v = b @ B
                    m2 = v.sum()
                    if m2 <= 1e-15:
                        continue
                    b2 = v / m2
                    key = (r2, tuple(np.round(b2, 12)))
                    if key in new:
                        new[key][1] += m * m2
                    else:
                        new[key] = [r2, m * m2, b2]
            layer = new
        for r, m, b in layer.values():
            yield self.H, r, b, m

    def evaluate(self, Gamma, lam=0.0):
        J_r = J_phi = 0.0
        size = 0
        for t, r, b, m in self.tree(Gamma):
            size += 1
            if t == self.H:
                J_phi += m * float(b @ self.acc[r])
            else:
                a = self.greedy(Gamma, t, r, b)
                J_r += m * self.w(t) * float(b @ self.Rw[r][:, a])
        return dict(J_phi=J_phi, J_r=J_r, L=J_r + lam * J_phi, tree=size)


def master(cols, c):
    phi = np.array([x['J_phi'] for x in cols])
    r = np.array([x['J_r'] for x in cols])
    best = lbs.best_pair(list(zip(phi, r)), c)
    if best is None:
        return None, None, None
    res = linprog([-c, 1.0], A_ub=np.column_stack([phi, -np.ones(len(phi))]), b_ub=-r,
                  bounds=[(0, None), (None, None)], method='highs')
    lam, mu = (float(res.x[0]), float(res.x[1])) if res.success else (0.0, float(r.max()))
    return best, lam, mu


def fixed_memory_seeds(task, model, start, seed, lam, warm, cache):
    import cpi
    pols = []
    for base in (cpi.initial_policy(task, model, start, seed), warm):
        pol = base.copy()
        J0 = cpi.evaluate(task, model, pol)
        cpi.converge(task, model, pol, 'lagrange', dict(phi=J0['J_phi'], r=J0['J_r']), dict(alpha=0.9), [], 'L',
                     lam=lam)
        pols.append(pol)
    cache['sweeps'] = cache.get('sweeps', 0)
    return pols


def run(task, model, c, seed=0, start='dirichlet', max_iter=25, tol=1e-7, seeded=True, **solver):
    import cpi
    t0 = time.time()
    rng = np.random.default_rng(seed)
    bm = BeliefModel(task, model)
    R = lbs.reward_span(task)
    room = task['env']['info'].get('ceiling', 1.0) - c
    lam_hi = 2.0 * R / room if room > 0 else 1e6
    warm = cpi.initial_policy(task, model, start, seed)
    J0 = cpi.evaluate(task, model, warm)
    cpi.converge(dict(task, objective='P1'), model, warm, 'sat', dict(phi=J0['J_phi'], r=J0['J_r']), dict(alpha=0.9),
                 [], 'P1', rng=np.random.default_rng(seed) if start == 'copycoin' else None)
    cols, log, cache = [], [], {}

    def column(lam, tag):
        seeds = [bm.seed_alphas(task, model, p, lam) for p in fixed_memory_seeds(task, model, start, seed, lam, warm, cache)] \
            if seeded else []
        G = bm.solve(lam, rng, seeds=seeds, **solver)
        e = bm.evaluate(G, lam)
        col = dict(lam=lam, J_phi=e['J_phi'], J_r=e['J_r'], tree=e['tree'], tag=tag, secs=round(time.time() - t0, 2))
        cols.append(col)
        return col

    column(0.0, 'reward')
    column(lam_hi, 'satisfaction')
    status, it = 'max_iter', 0
    for it in range(max_iter):
        best, lam, mu = master(cols, c)
        if best is None:
            status = 'infeasible'
            break
        col = column(lam, f'column{it}')
        rc = col['J_r'] + lam * col['J_phi'] - mu
        nb, _, _ = master(cols, c)
        log.append(dict(iter=it, lam=lam, mu=mu, J_phi=col['J_phi'], J_r=col['J_r'], reduced_cost=rc,
                        value=nb[0], secs=round(time.time() - t0, 2)))
        if rc <= tol * (1 + abs(mu)):
            status = 'converged'
            break
    best, lam, mu = master(cols, c)
    rec = dict(method='cgcp_belief', c=c, status=status, iterations=it + 1, columns=len(cols), lam=lam, seeded=seeded,
               secs=round(time.time() - t0, 2), lam_hi=lam_hi)
    if best is not None:
        v, i, j, th = best
        rec.update(J_r=float(v), J_phi=float(th * cols[j]['J_phi'] + (1 - th) * cols[i]['J_phi']), theta=float(th),
                   pair=(cols[i]['J_phi'], cols[i]['J_r'], cols[j]['J_phi'], cols[j]['J_r']), mixed=bool(i != j))
    return rec, dict(columns=cols, iterations=log)
