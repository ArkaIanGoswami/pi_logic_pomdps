"""Direct search over small classes of agent-state policies, used by the tests to check exact values."""

import numpy as np
from scipy.optimize import minimize
import cpi


def reachable_slots(task, model):
    H, Z, A = task['horizon'], task['memory_size'], model['A']
    R = len(model['rows'])
    uniform = np.zeros((H, R, Z, A, Z))
    legal = model['legal']
    for r in range(R):
        uniform[:, r, :, legal[r], :] = 1.0 / (legal[r].sum() * Z)
    mus = cpi.occupancies(task, model, uniform)
    G = cpi.aux(model, Z)['G']
    slots = []
    for t in range(H):
        mass = G @ mus[t]
        for r, z in np.argwhere(mass > 1e-12):
            idx = [a * Z + w for a in np.flatnonzero(legal[r]) for w in range(Z)]
            slots.append((t, int(r), int(z), idx))
    return slots


def policy_from(theta, slots, shape):
    pol = np.zeros(shape)
    flat = pol.reshape(shape[0], shape[1], shape[2], -1)
    k = 0
    for t, r, z, idx in slots:
        x = theta[k:k + len(idx)]
        k += len(idx)
        e = np.exp(x - x.max())
        flat[t, r, z, idx] = e / e.sum()
    return pol


def search(task, model, c, restarts=50, seed=0, maxiter=300, scale=3.0):
    H, Z, A = task['horizon'], task['memory_size'], model['A']
    shape = (H, len(model['rows']), Z, A, Z)
    slots = reachable_slots(task, model)
    n = sum(len(s[3]) for s in slots)
    rng = np.random.default_rng(seed)

    def J(theta):
        e = cpi.evaluate(task, model, policy_from(theta, slots, shape))
        return e['J_phi'], e['J_r']

    best, ends = -np.inf, []
    for _ in range(restarts):
        x0 = rng.normal(scale=scale, size=n)
        res = minimize(lambda th: -J(th)[1], x0, method='SLSQP',
                       constraints=[dict(type='ineq', fun=lambda th: J(th)[0] - c)],
                       options=dict(maxiter=maxiter, ftol=1e-12))
        phi, r = J(res.x)
        ends.append((phi, r))
        if phi >= c - 1e-7:
            best = max(best, r)
    return best, ends, n
