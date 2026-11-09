"""A simulator with reset, step and rollouts of a controller."""

import numpy as np
from policies import probs


def _draw(rng, outcomes, weight):
    i = int(rng.choice(len(outcomes), p=np.array([weight(o) for o in outcomes]) / sum(map(weight, outcomes))))
    return outcomes[i]


def reset(task, rng):
    env, dfa = task['env'], task['dfa']
    s, y, _ = _draw(rng, env['initial'], lambda o: o[2])
    q = dfa['next'](dfa['initial'], env['labels'](y))
    z = int(rng.choice(task['memory_size'], p=task['initial_memory']))
    return s, y, q, z


def step(task, s, a, rng):
    sp, yp, _, r = _draw(rng, task['env']['step'](s, a), lambda o: o[2])
    return sp, yp, r


def sample(task, s, a, m, rng):
    outcomes = task['env']['step'](s, a)
    p = np.array([o[2] for o in outcomes])
    counts = np.bincount(rng.choice(len(outcomes), size=m, p=p / p.sum()), minlength=len(outcomes))
    out = {}
    for (sp, yp, _, r), k in zip(outcomes, counts):
        if k:
            out[sp, yp] = (out.get((sp, yp), (0, r))[0] + int(k), r)
    return out


def rollout(task, policy, rng, stop_at_terminal=True):
    env, dfa = task['env'], task['dfa']
    Z = task['memory_size']
    s, y, q, z = reset(task, rng)
    out = []
    for t in range(task['horizon']):
        p = probs(task, policy, t, y, q, z)
        a, zn = divmod(int(rng.choice(p.size, p=p.ravel())), Z)
        sp, yp, r = step(task, s, a, rng)
        qp = dfa['next'](q, env['labels'](yp))
        out.append(dict(t=t, s=s, y=y, q=q, z=z, a=a, z_next=zn, r=r,
                        s_next=sp, y_next=yp, q_next=qp, labels=sorted(env['labels'](yp))))
        s, y, q, z = sp, yp, qp, zn
        if stop_at_terminal and env['terminal'](s):
            break
    return out


def episode_summary(task, episode):
    g = task['gamma']
    last = episode[-1]
    return dict(steps=len(episode), accepted=last['q_next'] in task['dfa']['accepting'],
                rejected=last['q_next'] in task['dfa']['rejecting'],
                discounted_return=sum(g ** e['t'] * e['r'] for e in episode))
