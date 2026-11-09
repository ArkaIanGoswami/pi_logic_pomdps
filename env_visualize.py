"""Figures of the environments - Entirely Claude-generated (but verified)"""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from tasks import make_task

SIDE = ('#3b6fb6', '#d9822b')
TILE, EDGE, INK = '#e8e8e8', '#9a9a9a', '#222222'
ROCK = {None: '#8c8c8c', True: '#3a9a5b', False: '#c0443b'}


def _letter(ax, x, y, s, color=INK, size=13):
    ax.text(x, y, s, ha='center', va='center', fontsize=size, color=color, fontweight='bold')


def draw_tmaze(ax, task, state=None, show_cue=False):
    env = task['env']
    lay, cfg = env['layout'], env['config']
    flip = cfg['orientation'] == 'horizontal'
    xy = (lambda c: (c[1], -c[0])) if flip else (lambda c: c)
    cue, pos, held, opened, _ = state if state is not None else (None, None, -1, -1, 0)
    for c in lay['cells']:
        x, y = xy(c)
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=TILE, edgecolor=EDGE, lw=1))
    x, y = xy(lay['cue'])
    if show_cue and cue is not None:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=SIDE[cue], alpha=.35, lw=0))
    _letter(ax, x, y, 'C')
    _letter(ax, *xy(lay['junction']), 'J')
    for i in (0, 1):
        if held != i:
            _letter(ax, *xy(lay['keys'][i]), 'K', SIDE[i])
        x, y = xy(lay['doors'][i])
        ax.add_patch(Rectangle((x - .45, y - .45), .9, .9, facecolor=SIDE[i] if opened != i else 'none',
                               alpha=.55, edgecolor=SIDE[i], lw=2))
        _letter(ax, x, y, 'D', INK if opened != i else SIDE[i])
        _letter(ax, *xy(lay['goals'][i]), 'G', SIDE[i])
    if pos is not None:
        x, y = xy(pos)
        ax.add_patch(Circle((x + .3, y - .3), .13, color=INK, zorder=5))
        if held >= 0:
            ax.add_patch(Rectangle((x + .17, y + .17), .2, .2, color=SIDE[held], zorder=6))
    pts = [xy(c) for c in lay['cells']]
    ax.set_xlim(min(p[0] for p in pts) - .7, max(p[0] for p in pts) + .7)
    ax.set_ylim(min(p[1] for p in pts) - .7, max(p[1] for p in pts) + .7)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_rocksample(ax, task, state=None, checked=None, sampled=None):
    lay = task['env']['layout']
    W, H = lay['width'], lay['height']
    for x in range(1, W + 1):
        for y in range(1, H + 1):
            ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=TILE, edgecolor=EDGE, lw=1))
    ax.add_patch(Rectangle((W + .5, .5), .8, H, facecolor='#d7ead9', edgecolor=EDGE, lw=1))
    _letter(ax, W + .9, (H + 1) / 2, 'X')
    sx, sy = lay['start']
    _letter(ax, sx - .3, sy + .3, 'S', size=9)
    good = state[1] if state is not None else None
    sampled = sampled or {}
    for i, (x, y) in enumerate(lay['rocks']):
        if i in sampled:
            ax.add_patch(Circle((x, y), .3, facecolor='white', edgecolor=ROCK[sampled[i]], lw=3, zorder=2))
            _letter(ax, x, y, chr(ord('A') + i), ROCK[sampled[i]], 11)
            continue
        ax.add_patch(Circle((x, y), .32, color=ROCK[None if good is None else bool(good[i])], zorder=2))
        _letter(ax, x, y, chr(ord('A') + i), 'white', 11)
    if state is not None:
        pos = state[0]
        x, y = (W + .9, (H + 1) / 2 - .6) if pos == (-1, -1) else pos
        if checked is not None and pos != (-1, -1):
            rx, ry = lay['rocks'][checked]
            ax.plot([x, rx], [y, ry], ls='--', lw=1.5, color=INK, zorder=3)
        ax.add_patch(Circle((x + .25, y - .25), .14, color=INK, zorder=5))
    ax.set_xlim(.3, W + 1.5)
    ax.set_ylim(.3, H + .7)
    ax.set_aspect('equal')
    ax.axis('off')


def draw(ax, task, state=None, **kw):
    (draw_tmaze if task['family'] == 'tmaze' else draw_rocksample)(ax, task, state, **kw)


def figsize(task, scale=.55):
    if task['family'] == 'rocksample':
        lay = task['env']['layout']
        return (scale * (lay['width'] + 1.2), scale * lay['height'])
    cells = task['env']['layout']['cells']
    w = max(c[0] for c in cells) - min(c[0] for c in cells) + 1.4
    h = max(c[1] for c in cells) - min(c[1] for c in cells) + 1.4
    if task['env']['config']['orientation'] == 'horizontal':
        w, h = h, w
    return (scale * w, scale * h)


def parse_sets(items):
    out = {}
    for item in items or []:
        k, v = item.split('=', 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tasks', nargs='+')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--set', nargs='*', help='environment overrides, key=value')
    ap.add_argument('--outdir', default='figures')
    ap.add_argument('--format', default='png')
    a = ap.parse_args(argv)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    for name in a.tasks:
        task = make_task(name, seed=a.seed, **parse_sets(a.set))
        fig, ax = plt.subplots(figsize=figsize(task))
        draw(ax, task)
        out = Path(a.outdir) / f"{Path(name).stem}_seed{a.seed}.{a.format}"
        fig.savefig(out, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(out)


if __name__ == '__main__':
    main()
