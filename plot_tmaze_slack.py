"""Share of memoryless (P1) runs on the T-maze that reach J_phi = 1, against the slack - Entirely Claude-generated (but verified)"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

LENGTHS = (3, 5, 10, 20)
ALPHA = 0.9

COLORS = {
    'L': {3: '#b7d0f2', 5: '#86afe8', 10: '#4f8bdc', 20: '#1f5fb4'},
    'pooled': '#52514e',
    'lazy': '#eb6834',
    'exists': '#52514e',
    'text': '#0b0b0b', 'text_muted': '#52514e', 'axes': '#8a8984', 'grid': '#e4e3df',
}
LABELS = {
    'title': r'$|Z|=1$, margin 1',
    'xlabel': r'slack $W$',
    'ylabel': r'share of runs with $J_\varphi=1$',
    'L': r'$L={L}$',
    'pooled': r'all $L$',
    'lazy': r'all $L$, $\alpha=1$ (lazy)',
}
ANNOTATION = {
    'text': '',
    'dx': -0.2,
    'y': 0.64,
    'rotation': 90, 'ha': 'right', 'va': 'center', 'fontsize': 6,
}
LINES = {
    'L': dict(lw=1.3, marker='o', ms=3.0, mec='white', mew=0.5),
    'pooled': dict(lw=1.0, ls='--'),
    'lazy': dict(lw=1.0, ls=':'),
    'exists': dict(lw=0.8, ls=':'),
}
SHOW = {'pooled': True, 'lazy': True, 'exists': True, 'title': True}
LEGEND = dict(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6, columnspacing=1.0,
              handlelength=1.8, frameon=False)
FONT = dict(size=8, title=8.5, label=8, ticks=7)
YLIM = (-0.03, 1.05)


def style(scale):
    plt.rcParams.update({
        'font.size': FONT['size'] * scale, 'axes.titlesize': FONT['title'] * scale,
        'axes.labelsize': FONT['label'] * scale, 'xtick.labelsize': FONT['ticks'] * scale,
        'ytick.labelsize': FONT['ticks'] * scale, 'axes.edgecolor': COLORS['axes'], 'axes.labelcolor': COLORS['text'],
        'xtick.color': COLORS['text_muted'], 'ytick.color': COLORS['text_muted'], 'text.color': COLORS['text'],
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': COLORS['grid'],
        'grid.linewidth': 0.6, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def load(root):
    runs, route = {}, {}
    for f in sorted((Path(root) / 'tmaze_slack').glob('*/result.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        for r in d['records']:
            if d['job']['kind'] == 'cpi':
                runs.setdefault((r['L'], r['alpha'], r['slack']), []).append(r['J_phi'] >= 1 - 1e-9)
            elif d['job']['kind'] == 'route':
                route.setdefault((r['L'], r['slack']), []).append(bool(r.get('route_exists')))
    share = {k: float(np.mean(v)) for k, v in runs.items()}
    exists = {k: (sum(v), len(v)) for k, v in route.items()}
    return share, exists


def save(fig, out, name, dpi):
    for ext, kw in (('pdf', {}), ('png', dict(dpi=dpi))):
        path = out / f'{name}.{ext}'
        try:
            fig.savefig(path, **kw)
        except OSError:
            path = out / f'{name}_new.{ext}'
            fig.savefig(path, **kw)
        print('wrote', path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='runs_paper', help='folder holding tmaze_slack/')
    ap.add_argument('--out', default='reports_paper')
    ap.add_argument('--name', default='figA_tmaze_slack')
    ap.add_argument('--width', type=float, default=3.4, help='inches')
    ap.add_argument('--height', type=float, default=2.6, help='inches')
    ap.add_argument('--dpi', type=int, default=300, help='PNG resolution')
    ap.add_argument('--font-scale', type=float, default=1.0)
    a = ap.parse_args(argv)

    share, exists = load(a.root)
    if not share:
        raise SystemExit(f'no runs under {a.root}/tmaze_slack (run: python suite.py --result tmaze)')
    Ls = [L for L in LENGTHS if any(k[0] == L for k in share)]
    Ws = sorted({k[2] for k in share})
    full = [W for W in Ws if all(exists.get((L, W), (0, 0))[1] > 0 and
                                 exists[(L, W)][0] == exists[(L, W)][1] for L in Ls)]
    w0 = min(full) if full else None

    style(a.font_scale)
    fig, ax = plt.subplots(figsize=(a.width, a.height))
    for L in Ls:
        ax.plot(Ws, [share.get((L, ALPHA, W), np.nan) for W in Ws], color=COLORS['L'].get(L, COLORS['pooled']),
                label=LABELS['L'].format(L=L), **LINES['L'])
    if SHOW['pooled']:
        ax.plot(Ws, [np.mean([share.get((L, ALPHA, W), np.nan) for L in Ls]) for W in Ws], color=COLORS['pooled'],
                label=LABELS['pooled'], **LINES['pooled'])
    if SHOW['lazy']:
        ax.plot(Ws, [np.mean([share.get((L, 1.0, W), np.nan) for L in Ls]) for W in Ws], color=COLORS['lazy'],
                label=LABELS['lazy'], **LINES['lazy'])
    if SHOW['exists'] and w0 is not None:
        ax.axvline(w0, color=COLORS['exists'], **LINES['exists'])
        if ANNOTATION['text']:
            ax.text(w0 + ANNOTATION['dx'], ANNOTATION['y'], ANNOTATION['text'], color=COLORS['text_muted'],
                    rotation=ANNOTATION['rotation'], ha=ANNOTATION['ha'], va=ANNOTATION['va'],
                    fontsize=ANNOTATION['fontsize'] * a.font_scale)
    ax.set_xlabel(LABELS['xlabel'])
    ax.set_ylabel(LABELS['ylabel'])
    ax.set_ylim(*YLIM)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if SHOW['title']:
        ax.set_title(LABELS['title'], loc='left')
    ax.legend(**dict(LEGEND, fontsize=LEGEND['fontsize'] * a.font_scale))
    fig.tight_layout()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, a.name, a.dpi)

    print('\nSuccessfulli = 1 (alpha = %g); last row: layouts where the route policy reaches J_phi = 1' % ALPHA)
    print('W      ' + ''.join(f'{W:>9d}' for W in Ws))
    for L in Ls:
        print(f'L={L:<4d} ' + ''.join(f"{share.get((L, ALPHA, W), np.nan):9.2f}" for W in Ws))
    cells = [f"{sum(exists.get((L, W), (0, 0))[0] for L in Ls)}/{sum(exists.get((L, W), (0, 0))[1] for L in Ls)}" for W in Ws]
    print('route  ' + ''.join(f'{c:>9s}' for c in cells))


if __name__ == '__main__':
    main()
