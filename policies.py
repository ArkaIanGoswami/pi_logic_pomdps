"""Finite-memory controllers as functions."""

import hashlib
import pickle
import numpy as np


def probs(task, policy, t, y, q, z):
    A, Z = len(task['env']['actions']), task['memory_size']
    p = np.array(policy(t, y, q, z), dtype=float).reshape(A, Z)
    legal = np.zeros(A, bool)
    legal[list(task['env']['legal'](y))] = True
    p[~legal] = 0.0
    if p.sum() <= 0:
        p[legal] = 1.0
    return p / p.sum()


def uniform_policy(task):
    A, Z = len(task['env']['actions']), task['memory_size']
    return lambda t, y, q, z: np.ones((A, Z))


def dirichlet_row(seed, key, n_actions, memory_size, concentration=1.0):
    h = int.from_bytes(hashlib.blake2b(repr((int(seed), key)).encode(), digest_size=8).digest(), 'little')
    rng = np.random.default_rng(h)
    return rng.dirichlet(np.full(n_actions * memory_size, concentration)).reshape(n_actions, memory_size)


def random_policy(task, seed=0):
    A, Z = len(task['env']['actions']), task['memory_size']
    return lambda t, y, q, z: dirichlet_row(seed, (int(t), y, q, int(z)), A, Z)


def table_policy(task, table, seed=None):
    fallback = random_policy(task, seed) if seed is not None else uniform_policy(task)

    def pi(t, y, q, z):
        row = table.get((int(t), y, q, int(z)))
        return fallback(t, y, q, z) if row is None else row
    return pi


def save_policy(path, table, memory_size, seed=0):
    with open(path, 'wb') as f:
        pickle.dump({'table': table, 'memory_size': int(memory_size), 'seed': int(seed)}, f)


def load_policy(task, path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    if int(d['memory_size']) != task['memory_size']:
        raise ValueError(f"policy has |Z|={d['memory_size']}, task has |Z|={task['memory_size']}")
    return table_policy(task, d['table'], d.get('seed'))
