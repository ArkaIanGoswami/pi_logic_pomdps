"""Product model"""

from collections import defaultdict, deque
import numpy as np
from scipy import sparse
from policies import probs


def compile_product(task, max_nodes=3_000_000):
    env, dfa = task['env'], task['dfa']
    A = len(env['actions'])
    index, nodes, queue = {}, [], deque()

    def add(node):
        j = index.get(node)
        if j is None:
            if len(nodes) >= max_nodes:
                raise RuntimeError(f'more than {max_nodes} product nodes')
            j = index[node] = len(nodes)
            nodes.append(node)
            queue.append(node)
        return j

    init = defaultdict(float)
    for s, y, p in env['initial']:
        init[add((s, y, dfa['next'](dfa['initial'], env['labels'](y))))] += p
    trip = [([], [], []) for _ in range(A)]
    reward = defaultdict(float)
    while queue:
        s, y, q = node = queue.popleft()
        j = index[node]
        term = env['terminal'](s)
        for a in range(A):
            r_, c_, v_ = trip[a]
            if term:
                r_.append(j)
                c_.append(j)
                v_.append(1.0)
                continue
            for sp, yp, p, r in env['step'](s, a):
                i = add((sp, yp, dfa['next'](q, env['labels'](yp))))
                r_.append(j)
                c_.append(i)
                v_.append(p)
                if r:
                    reward[j, a] += p * r
    n = len(nodes)
    P = [sparse.csr_matrix((v, (r, c)), shape=(n, n)) for r, c, v in trip]
    R = np.zeros((n, A))
    for (j, a), v in reward.items():
        R[j, a] = v
    initial = np.zeros(n)
    for j, p in init.items():
        initial[j] = p
    row_index, row_of_node = {}, np.empty(n, dtype=np.int64)
    for j, (s, y, q) in enumerate(nodes):
        row_of_node[j] = row_index.setdefault((y, q), len(row_index))
    rows = list(row_index)
    legal = np.zeros((len(rows), A), dtype=bool)
    for k, (y, q) in enumerate(rows):
        legal[k, list(env['legal'](y))] = True
    return dict(nodes=nodes, index=index, P=P, R=R, initial=initial, rows=rows,
                row_of_node=row_of_node, legal=legal, n=n, A=A,
                accepting=np.array([q in dfa['accepting'] for _, _, q in nodes]),
                rejecting=np.array([q in dfa['rejecting'] for _, _, q in nodes]),
                terminal=np.array([env['terminal'](s) for s, _, _ in nodes]))


def row_probs(task, model, policy, t):
    Z = task['memory_size']
    return np.array([[probs(task, policy, t, y, q, z) for z in range(Z)] for y, q in model['rows']])


def evaluate(task, model, policy):
    Z, H, g = task['memory_size'], task['horizon'], task['gamma']
    mu = model['initial'][:, None] * np.asarray(task['initial_memory'])[None, :]
    J_r = 0.0
    for t in range(H):
        rows = row_probs(task, model, policy, t)[model['row_of_node']]
        act = np.einsum('nz,nzaw->naw', mu, rows)
        J_r += g ** t * float(np.sum(act.sum(axis=2) * model['R']))
        new = np.zeros_like(mu)
        for a in range(model['A']):
            for w in range(Z):
                v = act[:, a, w]
                if v.any():
                    new[:, w] += model['P'][a].T @ v
        mu = new
    return dict(J_phi=float(mu[model['accepting']].sum()),
                J_reject=float(mu[model['rejecting']].sum()), J_r=J_r)
