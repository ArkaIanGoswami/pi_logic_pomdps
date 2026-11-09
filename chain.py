"""The chain environment of Example 1, its labels and its automaton."""

import numpy as np
from logic import make_dfa


def actions():
    return ('a0', 'a1')


def defaults():
    return dict(decisions=2, gamma=1.0)


def v2(T, c):
    return float((1.0 - c ** (1.0 / T)) ** T)


def v2_bar(c):
    return float(1.0 - c)


def max_lagrangian(lam):
    return float(max(1.0, lam))


def make_env(cfg):
    cfg = {**defaults(), **cfg}
    T = int(cfg['decisions'])
    if T < 1:
        raise ValueError('decisions must be >= 1')

    def step(state, a):
        k, cls = state
        if k >= T:
            return [(state, 'none', 1.0, 0.0)]
        if k == 0:
            new = 'all1' if a == 1 else 'all0'
        else:
            new = cls if (cls, a) in (('all1', 1), ('all0', 0)) else 'mixed'
        s2 = (k + 1, new)
        if k + 1 < T:
            return [(s2, 'none', 1.0, 0.0)]
        return [(s2, 'sat' if new == 'all1' else 'none', 1.0, 1.0 if new == 'all0' else 0.0)]

    return dict(name=f'chain{T}', config=cfg, layout=dict(decisions=T), actions=actions(), horizon=T,
                gamma=float(cfg['gamma']), initial=[((0, 'none'), 'none', 1.0)], step=step,
                labels=lambda y: frozenset({'goal'}) if y == 'sat' else frozenset(),
                legal=lambda y: (0, 1), terminal=lambda s: s[0] >= T,
                reward_range=(0.0, 1.0), info=dict(decisions=T, ceiling=1.0, reward_span=1.0),
                max_lagrangian=lambda lam, Z: max_lagrangian(lam))


def make_spec():
    return make_dfa(('q0', 'accept'), 0, {1}, (), lambda q, lab: 1 if 'goal' in lab else q, 'F goal')


def check_values(T=3, c=0.5, grid=201):
    xs = np.linspace(0, 1, grid)
    best = 0.0
    for x in xs:
        if x ** T >= c - 1e-12:
            best = max(best, (1 - x) ** T)
    return best, v2(T, c)
