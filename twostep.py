"""The two-step environment of Example 1, its labels and its automaton."""

from logic import make_dfa


def actions():
    return ('a0', 'a1')


def defaults():
    return dict(horizon=2, gamma=1.0)


def make_env(cfg):
    cfg = {**defaults(), **cfg}

    def step(state, a):
        if state == 'start':
            return [(('mid', a), 'mid', 1.0, 0.0)]
        if state[0] == 'mid':
            first = state[1]
            obs = 'sat' if (first, a) == (1, 1) else 'none'
            return [(('end', first, a), obs, 1.0, 1.0 if (first, a) == (0, 0) else 0.0)]
        return [(state, 'none', 1.0, 0.0)]

    return dict(name='twostep', config=cfg, layout={}, actions=actions(), horizon=int(cfg['horizon']),
                gamma=float(cfg['gamma']), initial=[('start', 'start', 1.0)], step=step,
                labels=lambda y: frozenset({'goal'}) if y == 'sat' else frozenset(),
                legal=lambda y: (0, 1), terminal=lambda s: s != 'start' and s[0] == 'end',
                reward_range=(0.0, 1.0), info=dict(v2=0.25, v2_bar=0.75))


def make_spec():
    return make_dfa(('q0', 'accept'), 0, {1}, (), lambda q, lab: 1 if 'goal' in lab else q, 'F goal')
