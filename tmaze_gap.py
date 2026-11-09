"""The polar maze, its labels and its automaton."""

from math import comb
from scipy.optimize import minimize_scalar
from logic import make_dfa


def actions():
    return ('forward', 'scan', 'left', 'right')


def action_index(name):
    return actions().index(name)


def defaults():
    return dict(surveys=2, budget=0, corridor_length=None, bonus=1.0, gamma=1.0)


def survey_tiles(K, L):
    return tuple(int(round(i * L / (K + 1))) for i in range(1, K + 1))


def make_env(cfg):
    cfg = {**defaults(), **cfg}
    K, s = int(cfg['surveys']), int(cfg['budget'])
    L = int(cfg['corridor_length'] or 2 * K + 1)
    if K < 1 or not 0 <= s <= K or L < K + 1:
        raise ValueError('need surveys >= 1, 0 <= budget <= surveys, corridor_length >= surveys + 1')
    tiles = survey_tiles(K, L)
    if len(set(tiles)) != K or min(tiles) < 1 or max(tiles) > L - 1:
        raise ValueError(f'survey tiles {tiles} must be distinct and inside 1..{L - 1}')
    survey = frozenset(tiles)
    forward, scan, left, right = (action_index(a) for a in actions())

    def step(state, a):
        cue, pos, scans, done = state
        if done:
            return [(state, (pos, 'end'), 1.0, 0.0)]
        if pos < L:
            n = scans + int(a == scan and pos in survey)
            s2 = (cue, pos + 1, n, 0)
            return [(s2, (pos + 1, 'none'), 1.0, 0.0)]
        side = 0 if a == left else 1
        good = side == cue and scans <= s
        bonus = float(cfg['bonus']) if scans == K else 0.0
        return [((cue, L + 1 + side, scans, 1), (L + 1 + side, 'good' if good else 'bad'), 1.0, bonus)]

    def legal(y):
        pos, token = y
        if token in ('good', 'bad', 'end'):
            return tuple(range(len(actions())))
        if pos >= L:
            return (left, right)
        return (forward, scan) if pos in survey else (forward,)

    def labels(y):
        return frozenset({'good': ('g_c',), 'bad': ('fail',)}.get(y[1], ()))

    initial = [((c, 0, 0, 0), (0, f'cue{c}'), 0.5) for c in (0, 1)]
    return dict(name='tmaze_gap', config=cfg, layout=dict(corridor_length=L, surveys=tiles, budget=s),
                actions=actions(), horizon=L + 1, gamma=float(cfg['gamma']), initial=initial, step=step,
                labels=labels, legal=legal, terminal=lambda st: bool(st[3]),
                reward_range=(0.0, float(cfg['bonus'])),
                info=dict(surveys=K, budget=s, corridor_length=L, ceiling=1.0, reward_span=float(cfg['bonus'])),
                max_lagrangian=(lambda lam, Z: float(cfg['bonus']) * max_lagrangian(lam / float(cfg['bonus']), Z))
                if s < K else None)


def make_spec():
    def step(q, lab):
        if 'fail' in lab:
            return 2
        if 'g_c' in lab:
            return 1
        return q
    return make_dfa(('q0', 'accept', 'reject'), 0, {1}, {2}, step, 'F g_c')


def v2_bar(c, K=None, s=None):
    if K is not None and s is not None and s >= K:
        return 1.0
    return float(1.0 - c)


def max_lagrangian(lam, Z=2):
    return float(max(1.0, lam if Z >= 2 else lam / 2))


def _branch_frontier(K, s, phi):
    if phi <= 0:
        return 1.0

    def p_le(y):
        return sum(comb(K, j) * y ** j * (1 - y) ** (K - j) for j in range(s + 1))

    if p_le(1.0) >= phi:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if p_le(mid) >= phi:
            lo = mid
        else:
            hi = mid
    return lo ** K


def v2_independent(c, K, s):
    return _branch_frontier(K, s, c)


def v2(c, K, s):
    if s >= K:
        return 1.0
    if c <= 0.5 or s == K - 1:
        return float(1.0 - c)
    if s > 0:
        return float('nan')
    lo = max(0.0, 2 * c - 1)
    hi = min(1.0, 2 * c)
    f = lambda p0: 0.5 * (_branch_frontier(K, s, p0) + _branch_frontier(K, s, 2 * c - p0))
    res = minimize_scalar(lambda p0: -f(p0), bounds=(lo, hi), method='bounded', options=dict(xatol=1e-12))
    return float(max(-res.fun, f(lo), f(hi)))
