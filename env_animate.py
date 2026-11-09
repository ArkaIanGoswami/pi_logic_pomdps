"""GIF animations of single episodes of a controller - Entirely Claude-generated (but verified)"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cpi
from env_visualize import draw, figsize, parse_sets
from policies import load_policy, random_policy, uniform_policy
from rocksample import action_index, first_check
from simulate import rollout
from tasks import make_task


def frames(task, episode):
    if task['family'] == 'tmaze':
        return [(episode[0]['s'], dict(show_cue=True))] + [(e['s_next'], {}) for e in episode]
    rocks, sampled = task['env']['layout']['rocks'], {}
    check0, sample = first_check(), action_index('sample')
    out = [(episode[0]['s'], dict(sampled={}))]
    for e in episode:
        kw = dict(sampled=sampled)
        if e['a'] >= check0:
            kw['checked'] = e['a'] - check0
        elif e['a'] == sample and e['s'][0] in rocks:
            sampled = {**sampled, rocks.index(e['s'][0]): bool(e['s'][1][rocks.index(e['s'][0])])}
            kw['sampled'] = sampled
        out.append((e['s_next'], kw))
    return out


def captions(task, episode):
    g, out, ret = task['gamma'], [], 0.0
    out.append(f"t = 1   z = {episode[0]['z']}   return = 0.0")
    for e in episode:
        ret += g ** e['t'] * e['r']
        out.append(f"t = {e['t'] + 2}   z = {e['z_next']}   return = {ret:.1f}")
    ok = episode[-1]['q_next'] in task['dfa']['accepting']
    out[-1] += '   ' + ('\u03c6 holds' if ok else '\u03c6 fails')
    return out


def animate(task, episode, out, fps=3, hold=4, dpi=150, annotate=False):
    seq = frames(task, episode)
    text = captions(task, episode) if annotate else None
    seq += [seq[-1]] * hold
    w, h = figsize(task)
    fig, ax = plt.subplots(figsize=(max(w, 2.6), h + 0.35) if annotate else (w, h))
    if annotate:
        fig.subplots_adjust(top=1 - 0.35 / (h + 0.35))
        label = fig.text(0.5, 1 - 0.17 / (h + 0.35), '', ha='center', va='center', fontsize=7)

    def update(i):
        ax.clear()
        state, kw = seq[i]
        draw(ax, task, state, **kw)
        if annotate:
            label.set_text(text[min(i, len(text) - 1)])

    anim = FuncAnimation(fig, update, frames=len(seq), interval=1000 / fps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task')
    ap.add_argument('--variant', default=None, choices=('logic', 'free'))
    ap.add_argument('--seed', type=int, default=0, help='task seed (T-maze key layout)')
    ap.add_argument('--memory-size', type=int, default=None)
    ap.add_argument('--policy', default='random', help="pickle path, 'random' or 'uniform'")
    ap.add_argument('--episode-seed', type=int, default=0)
    ap.add_argument('--set', nargs='*', help='environment overrides, key=value')
    ap.add_argument('--fps', type=float, default=3)
    ap.add_argument('--out', default=None)
    ap.add_argument('--annotate', action='store_true', help='show time, memory state and return on every frame')
    a = ap.parse_args(argv)
    task = make_task(a.task, variant=a.variant, seed=a.seed, memory_size=a.memory_size, **parse_sets(a.set))
    if a.policy == 'random':
        pi = random_policy(task, a.seed)
    elif a.policy == 'uniform':
        pi = uniform_policy(task)
    elif a.policy.endswith('.npz'):
        saved = cpi.load_policy(a.policy)
        task = dict(task, memory_size=saved['pol'].shape[2], initial_memory=tuple(saved['initial_memory']))
        pi = cpi.as_function(saved)
    else:
        pi = load_policy(task, a.policy)
    episode = rollout(task, pi, np.random.default_rng(a.episode_seed))
    out = a.out or f"animations/{Path(a.task).stem}_seed{a.seed}_ep{a.episode_seed}.gif"
    print(animate(task, episode, out, a.fps, annotate=a.annotate))


if __name__ == '__main__':
    main()
