"""The RockSample environment, its labels and its automaton."""

from itertools import product
from math import ceil, exp, hypot, log
from logic import make_dfa


def sensor(name):
    return {'good': 1, 'bad': 2, 'none': 3}[name]


def base_actions():
    return ('sample', 'north', 'east', 'south', 'west')


def action_index(name):
    return base_actions().index(name)


def first_check():
    return len(base_actions())


def terminal_position():
    return (-1, -1)


def feedback_codes():
    return {'good_sample': 1, 'bad_sample': 2, 'exit': 3, 'near_check': 4}


def events():
    return {v: k for k, v in feedback_codes().items()}


def defaults():
    return dict(width=5, height=5, rocks=((1, 1), (3, 3), (4, 4)), start=(1, 1),
                horizon=32, sensor_efficiency=20.0, good_rock_reward=10.0,
                bad_rock_penalty=-10.0, exit_reward=10.0, step_penalty=0.0,
                sensor_use_penalty=0.0, gamma=0.95, p_good=0.5,
                condition_at_least_one_good=False, near_check_distance=0.0)


def ceiling(cfg):
    if cfg['condition_at_least_one_good']:
        return 1.0
    return 1.0 - (1.0 - cfg['p_good']) ** len(cfg['rocks'])


def strictest_delta(cfg):
    return max(0.1, ceil(10 * (1.0 - ceiling(cfg)) - 1e-9) / 10)


def make_env(cfg):
    cfg = {**defaults(), **cfg}
    W, Hgt = cfg['width'], cfg['height']
    rocks = [tuple(r) for r in cfg['rocks']]
    K = len(rocks)
    rock_at = {r: i for i, r in enumerate(rocks)}
    actions = base_actions() + tuple(f'check{i + 1}' for i in range(K))
    s_good, s_bad, s_none = sensor('good'), sensor('bad'), sensor('none')
    sample, north, east, south, west = (action_index(a) for a in base_actions())
    check0, term = first_check(), terminal_position()
    feedback, event_of = feedback_codes(), events()

    def token(s, event=None):
        return s + 3 * (feedback[event] if event else 0)

    def accuracy(pos, i):
        d = hypot(pos[0] - rocks[i][0], pos[1] - rocks[i][1])
        return 0.5 * (1.0 + exp(-d * log(2) / cfg['sensor_efficiency']))

    def step(state, a):
        pos, good = state
        if pos == term:
            return [(state, (term, s_none), 1.0, 0.0)]
        r = cfg['step_penalty']
        if a == sample:
            i = rock_at.get(pos)
            if i is None:
                return [(state, (pos, s_none), 1.0, r)]
            bits = list(good)
            bits[i] = False
            event = 'good_sample' if good[i] else 'bad_sample'
            r += cfg['good_rock_reward'] if good[i] else cfg['bad_rock_penalty']
            return [((pos, tuple(bits)), (pos, token(s_none, event)), 1.0, r)]
        if a < check0:
            x, y = pos
            x, y = {north: (x, y + 1), south: (x, y - 1), east: (x + 1, y), west: (x - 1, y)}[a]
            if x > W:
                return [((term, (False,) * K), (term, token(s_none, 'exit')), 1.0,
                         r + cfg['exit_reward'])]
            nxt = (max(1, min(W, x)), max(1, min(Hgt, y)))
            return [((nxt, good), (nxt, s_none), 1.0, r)]
        i = a - check0
        p = accuracy(pos, i)
        near = hypot(pos[0] - rocks[i][0], pos[1] - rocks[i][1]) <= cfg['near_check_distance']
        event = 'near_check' if near else None
        r += cfg['sensor_use_penalty']
        p_good = p if good[i] else 1.0 - p
        return [(state, (pos, token(s, event)), q, r)
                for s, q in ((s_good, p_good), (s_bad, 1.0 - p_good)) if q > 0]

    def labels(y):
        code = (y[1] - 1) // 3
        return frozenset((event_of[code],)) if code in event_of else frozenset()

    def legal(y):
        pos = y[0]
        if pos == term or pos in rock_at:
            return tuple(range(len(actions)))
        return tuple(range(1, len(actions)))

    p = cfg['p_good']
    norm = ceiling(cfg) if cfg['condition_at_least_one_good'] else 1.0
    initial = []
    for bits in product((False, True), repeat=K):
        if cfg['condition_at_least_one_good'] and not any(bits):
            continue
        w = p ** sum(bits) * (1 - p) ** (K - sum(bits)) / norm
        if w > 0:
            initial.append(((tuple(cfg['start']), bits), (tuple(cfg['start']), s_none), w))
    rmin = min(0.0, cfg['bad_rock_penalty'], cfg['step_penalty'], cfg['sensor_use_penalty'])
    rmax = max(cfg['good_rock_reward'], cfg['exit_reward'], 0.0) + max(0.0, cfg['step_penalty'])
    return dict(name='rocksample', config=cfg, layout=dict(width=W, height=Hgt, rocks=rocks,
                start=tuple(cfg['start'])), actions=actions, horizon=int(cfg['horizon']),
                gamma=float(cfg['gamma']), initial=initial, step=step, labels=labels, legal=legal,
                terminal=lambda s: s[0] == term, reward_range=(rmin, rmax),
                info=dict(ceiling=ceiling(cfg), strictest_delta=strictest_delta(cfg)))


def make_spec():
    def step(q, lab):
        if 'bad_sample' in lab:
            return 3
        if 'exit' in lab:
            return 2 if q == 1 else 3
        if 'good_sample' in lab:
            return 1
        return q
    return make_dfa(('q0', 'q1', 'accept', 'reject'), 0, {2}, {3}, step,
                    'F(good & F exit) & (!bad U exit)')
