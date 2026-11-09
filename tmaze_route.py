"""The memoryless T-maze policy that carries the cue in its route."""

import numpy as np
import cpi

FWD, BWD, LEFT, RIGHT, PICK, OPEN = range(6)


def _actions(layout, cue, delay, L, a):
    acts = [BWD, FWD] if delay else []
    key = layout['keys'][cue][1]
    for i in range(L):
        if i == key:
            acts.append(PICK)
        acts.append(FWD)
    side = LEFT if cue == 0 else RIGHT
    return acts + [FWD] + [side] * a + [OPEN, side, side]


def route_policy(task, model):
    if task['memory_size'] != 1:
        raise ValueError('the route policy is memoryless: build the task with memory_size=1')
    env, dfa = task['env'], task['dfa']
    lay, H = env['layout'], task['horizon']
    L, a = env['config']['corridor_length'], env['config']['branch_length']
    row = {r: i for i, r in enumerate(model['rows'])}
    want = {}
    for s, y, _ in env['initial']:
        cue = int(y[1][3:])
        q = dfa['next'](dfa['initial'], env['labels'](y))
        for t, act in enumerate(_actions(lay, cue, cue == 1, L, a)):
            if t >= H:
                return None, 'route longer than the horizon'
            r = row[(y, q)]
            if not model['legal'][r, act]:
                return None, 'back step not possible (no start margin)' if t == 0 else f'illegal action at t={t + 1}'
            if want.get((t, r), act) != act:
                return None, f'the two routes conflict at t={t + 1}'
            want[(t, r)] = act
            (s, y, _, _), = env['step'](s, act)
            q = dfa['next'](q, env['labels'](y))
            if env['terminal'](s):
                break
    pol = np.zeros((H, len(model['rows']), 1, model['A'], 1))
    for t in range(H):
        for r in range(len(model['rows'])):
            act = want.get((t, r), int(np.flatnonzero(model['legal'][r])[0]))
            pol[t, r, 0, act, 0] = 1.0
    return pol, 'ok'


def route_value(task, model):
    pol, status = route_policy(task, model)
    if pol is None:
        return None, status
    return cpi.evaluate(task, model, pol)['J_phi'], status
