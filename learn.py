"""CPI on a model learned from simulator queries."""

from collections import deque
from math import log, sqrt
import numpy as np
from scipy.stats import beta as beta_dist
import cpi
from product import compile_product
from simulate import sample


def learn_model(task, m, n_resets, rng):
    env = task['env']
    p = np.array([w for _, _, w in env['initial']])
    draws = np.bincount(rng.choice(len(p), size=n_resets, p=p / p.sum()), minlength=len(p))
    initial = [(s, y, k / n_resets) for (s, y, _), k in zip(env['initial'], draws) if k]
    queue, seen, kernel = deque(), set(), {}
    for s, _, _ in initial:
        if s not in seen:
            seen.add(s)
            queue.append(s)
    while queue:
        s = queue.popleft()
        if env['terminal'](s):
            continue
        for a in range(len(env['actions'])):
            counts = sample(task, s, a, m, rng)
            kernel[s, a] = [(sp, yp, k / m, r) for (sp, yp), (k, r) in counts.items()]
            for sp, _ in counts:
                if sp not in seen:
                    seen.add(sp)
                    queue.append(sp)
    learned = dict(task, env=dict(env, initial=initial, step=lambda s, a: kernel[s, a]))
    stats = dict(latent_states=len(seen), pairs=len(kernel), queries=m * len(kernel), resets=n_resets)
    return learned, kernel, stats


def _log_2k_minus_2(k):
    return k * log(2) + log(1 - 2.0 ** (1 - k))


def radii(task, kernel, m, n_resets, beta=0.05):
    env = task['env']
    N = len({s for s, _ in kernel}) * len(env['actions'])
    eps = {}
    for (s, a), outs in kernel.items():
        k = len(env['step'](s, a))
        if k == 1:
            eps[s, a] = 0.0
        elif len(outs) == 1:
            eps[s, a] = min(2.0, 2.0 / m * log(3 * k * N / beta))
        else:
            eps[s, a] = min(2.0, sqrt(2.0 / m * (log(3) + _log_2k_minus_2(k) + log(N) - log(beta))))
    K0 = len(env['initial'])
    eps_rho = 0.0 if K0 <= 1 else min(2.0, sqrt(2.0 / n_resets * (log(3) + _log_2k_minus_2(K0) - log(beta))))
    return eps, eps_rho


def delta_phi(task, model, pol, eps, eps_rho):
    undecided = ~(model['accepting'] | model['rejecting'])
    e = np.zeros((model['n'], model['A']))
    for j, (s, _, _) in enumerate(model['nodes']):
        for a in range(model['A']):
            e[j, a] = eps.get((s, a), 0.0)
    e *= 0.5 * undecided[:, None]
    mu, total = cpi.start_occupancy(task, model), 0.5 * eps_rho
    for t in range(task['horizon']):
        act = np.einsum('nz,nzaw->na', mu, pol[t][model['row_of_node']])
        total += float((act * e).sum())
        mu = cpi.forward(model, pol[t], mu)
    return total


def holdout(task, model, pol, n, rng, beta=0.05):
    if '_cum' not in model:
        cum = []
        for P in model['P']:
            C = P.tocsr().copy()
            for j in range(model['n']):
                lo, hi = C.indptr[j], C.indptr[j + 1]
                C.data[lo:hi] = np.cumsum(C.data[lo:hi])
            cum.append(C)
        model['_cum'] = cum
    cum, Z = model['_cum'], pol.shape[2]
    j = rng.choice(model['n'], size=n, p=model['initial'] / model['initial'].sum())
    z = rng.choice(Z, size=n, p=np.asarray(task['initial_memory'], float))
    ret = np.zeros(n)
    for t in range(task['horizon']):
        p = pol[t][model['row_of_node'][j], z].reshape(n, -1)
        choice = np.minimum((np.cumsum(p, axis=1) < rng.random(n)[:, None]).sum(axis=1), p.shape[1] - 1)
        a, z = choice // Z, choice % Z
        ret += task['gamma'] ** t * model['R'][j, a]
        u, nxt = rng.random(n), np.empty(n, dtype=np.int64)
        for ai in np.unique(a):
            sel = np.flatnonzero(a == ai)
            C, rows = cum[ai], j[sel]
            pos, end = C.indptr[rows].copy(), C.indptr[rows + 1] - 1
            for _ in range(int((end - pos).max(initial=0)) + 1):
                move = (C.data[pos] < u[sel]) & (pos < end)
                if not move.any():
                    break
                pos[move] += 1
            nxt[sel] = C.indices[pos]
        j = nxt
    s = int(model['accepting'][j].sum())
    lower = 0.0 if s == 0 else float(beta_dist.ppf(beta, s, n - s + 1))
    return s, lower, float(ret.mean())


def run(task, cfg, levels, seed, true_model=None):
    rng = np.random.default_rng(1_000_003 * seed + 7919 * cfg['m'] + 17)
    learned, kernel, stats = learn_model(task, cfg['m'], cfg.get('resets', 2000), rng)
    model_l = compile_product(learned)
    model_t = true_model or compile_product(task)
    fallback = cpi.initial_policy(task, model_t, cfg.get('start', 'dirichlet'), seed)

    def to_true(pol):
        return cpi.transfer(pol, model_l['rows'], model_t['rows'], fallback)

    def on_sweep(pol):
        J = cpi.evaluate(task, model_t, to_true(pol))
        return dict(true_J_phi=J['J_phi'], true_J_r=J['J_r'])

    plan = [(name, c_plan) for name, _, c_plan in levels]
    res = cpi.solve(learned, model_l, cfg, plan, rng=np.random.default_rng(seed),
                    on_sweep=on_sweep if cfg.get('track_true', True) else None)
    eps, eps_rho = radii(task, kernel, cfg['m'], cfg.get('resets', 2000), cfg.get('beta', 0.05))
    c_true = {name: c for name, c, _ in levels}
    records, policies = [], {}
    for name, r in res.items():
        pol_t = to_true(r['pol'])
        J = cpi.evaluate(task, model_t, pol_t)
        s, lower, ret = holdout(task, model_t, pol_t, cfg.get('holdout', 5000), np.random.default_rng(10_007 * seed + 3),
                                cfg.get('beta', 0.05))
        c = c_true.get(name)
        records.append(dict(level=name, c=c, c_plan=r.get('c'), feasible_start=r['feasible_start'],
                            J_phi=J['J_phi'], J_r=J['J_r'], model_J_phi=r['J']['phi'], model_J_r=r['J']['r'],
                            holdout_n=cfg.get('holdout', 5000), holdout_successes=s, holdout_lower=lower,
                            holdout_return=ret, certified=None if c is None else bool(lower >= c),
                            feasible=None if c is None else bool(J['J_phi'] >= c - 1e-12),
                            delta_phi=delta_phi(learned, model_l, r['pol'], eps, eps_rho),
                            sweeps=len(r['log']) - 1, **stats))
        policies[name] = (pol_t, r['log'])
    return records, policies, model_t
