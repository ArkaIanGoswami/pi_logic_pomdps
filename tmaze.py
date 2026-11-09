"""The T-maze environment, its labels and its automaton."""

import numpy as np
from logic import make_dfa


def actions():
    return ('forward', 'backward', 'left', 'right', 'pick', 'open')


def action_index(name):
    return actions().index(name)


def moves():
    return {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}


def end_tokens():
    return ('good', 'bad', 'blocked', 'end')


def defaults():
    return dict(corridor_length=5, branch_length=0, start_margin=1, slack=2,
                horizon=None, allow_noop=False, layout_seed=0, goal_reward=1.0,
                wrong_goal_reward=-1.0, blocked_reward=0.0, gamma=1.0,
                orientation='vertical')


def shortest_path(cfg):
    return cfg['corridor_length'] + cfg['branch_length'] + 5


def horizon(cfg):
    if cfg.get('horizon') is not None:
        return int(cfg['horizon'])
    return shortest_path(cfg) + int(cfg['slack'])


def slack(cfg):
    return horizon(cfg) - shortest_path(cfg)


def make_layout(cfg):
    L, a, m = cfg['corridor_length'], cfg['branch_length'], cfg['start_margin']
    if L < 3 or a < 0 or m < 0:
        raise ValueError('need corridor_length >= 3, branch_length >= 0, start_margin >= 0')
    rng = np.random.default_rng(cfg['layout_seed'])
    k0, k1 = (int(i) for i in rng.choice(np.arange(1, L), size=2, replace=False))
    top = L + 1
    stem = {(0, i) for i in range(-m, L + 1)}
    bar = {(j, top) for j in range(-(a + 2), a + 3)}
    return dict(cells=stem | bar, cue=(0, 0), junction=(0, L), centre=(0, top),
                keys={0: (0, k0), 1: (0, k1)},
                doors={0: (-(a + 1), top), 1: (a + 1, top)},
                goals={0: (-(a + 2), top), 1: (a + 2, top)})


def _target(lay, pos, a):
    dx, dy = moves()[a]
    nxt = (pos[0] + dx, pos[1] + dy)
    return nxt if nxt in lay['cells'] else None


def _adjacent_doors(lay, pos):
    return [i for i, d in lay['doors'].items() if abs(d[0] - pos[0]) + abs(d[1] - pos[1]) == 1]


def _key_here(lay, pos, held):
    return any(p == pos and held != i for i, p in lay['keys'].items())


def legal_actions(cfg, lay, y):
    pos, token, has_key, door_open = y
    if token in end_tokens() or cfg['allow_noop']:
        return tuple(range(len(actions())))
    out = [a for a in moves() if _target(lay, pos, a) is not None]
    if token in ('key',) and not has_key:
        out.append(action_index('pick'))
    if _adjacent_doors(lay, pos) and not door_open:
        out.append(action_index('open'))
    return tuple(out)


def make_env(cfg):
    cfg = {**defaults(), **cfg}
    lay = make_layout(cfg)
    H = horizon(cfg)
    if H < shortest_path(cfg):
        raise ValueError('horizon is shorter than the shortest accepting path')
    move_set, pick = moves(), action_index('pick')

    def obs(state, event=None):
        cue, pos, held, opened, done = state
        if event is None:
            event = 'end' if done else ('key' if _key_here(lay, pos, held) else 'none')
        return (pos, event, int(held >= 0), int(opened >= 0))

    def step(state, a):
        cue, pos, held, opened, done = state
        if done:
            return [(state, obs(state, 'end'), 1.0, 0.0)]
        if a in move_set:
            nxt = _target(lay, pos, a)
            if nxt is None:
                return [(state, obs(state), 1.0, 0.0)]
            for i, d in lay['doors'].items():
                if nxt == d and opened != i:
                    s2 = (cue, pos, held, opened, 1)
                    return [(s2, obs(s2, 'blocked'), 1.0, cfg['blocked_reward'])]
            for i, g in lay['goals'].items():
                if nxt == g:
                    s2 = (cue, nxt, held, opened, 1)
                    good = (i == cue)
                    r = cfg['goal_reward'] if good else cfg['wrong_goal_reward']
                    return [(s2, obs(s2, 'good' if good else 'bad'), 1.0, r)]
            s2 = (cue, nxt, held, opened, 0)
            return [(s2, obs(s2), 1.0, 0.0)]
        if a == pick:
            for i, p in lay['keys'].items():
                if p == pos and held < 0:
                    s2 = (cue, pos, i, opened, 0)
                    return [(s2, obs(s2, 'picked'), 1.0, 0.0)]
            return [(state, obs(state), 1.0, 0.0)]
        near = _adjacent_doors(lay, pos)
        if not near or opened >= 0:
            return [(state, obs(state), 1.0, 0.0)]
        if held in near:
            s2 = (cue, pos, held, held, 0)
            return [(s2, obs(s2, 'opened'), 1.0, 0.0)]
        s2 = (cue, pos, held, opened, 1)
        return [(s2, obs(s2, 'blocked'), 1.0, cfg['blocked_reward'])]

    def labels(y):
        token = y[1]
        return frozenset({'picked': ('k',), 'opened': ('d',), 'good': ('g_c',),
                          'bad': ('fail',), 'blocked': ('fail',)}.get(token, ()))

    initial = [((c, lay['cue'], -1, -1, 0), (lay['cue'], f'cue{c}', 0, 0), 0.5) for c in (0, 1)]
    rewards = (cfg['wrong_goal_reward'], cfg['blocked_reward'], 0.0, cfg['goal_reward'])
    return dict(name='tmaze', config=cfg, layout=lay, actions=actions(), horizon=H,
                gamma=float(cfg['gamma']), initial=initial, step=step, labels=labels,
                legal=lambda y: legal_actions(cfg, lay, y),
                terminal=lambda s: bool(s[4]),
                reward_range=(min(rewards), max(rewards)),
                info=dict(shortest_path=shortest_path(cfg), slack=H - shortest_path(cfg)))


def make_spec():
    def step(q, lab):
        if 'fail' in lab:
            return 4
        if q == 0 and 'k' in lab:
            return 1
        if q == 1 and 'd' in lab:
            return 2
        if q == 2 and 'g_c' in lab:
            return 3
        return q
    return make_dfa(('q0', 'qK', 'qKD', 'qF', 'qX'), 0, {3}, {4}, step,
                    'F(k & F(d & F g_c))')
