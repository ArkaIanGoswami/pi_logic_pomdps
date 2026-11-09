"""Tables, figures and in-text numbers for (P2) on RockSample and the polar maze - Entirely Claude-generated (but verified)"""

import argparse
import json
import warnings
from collections import defaultdict
from math import comb
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

SERIES = (
    ('cpi2', r'CPI, $|\mathcal{Z}|=2$', '#2a78d6', '-'),
    ('cpi4', r'CPI, $|\mathcal{Z}|=4$', '#2a78d6', '--'),
    ('npg', 'NPG-PD', '#8e44ad', '-'),
    ('lbs', 'LBS', '#1baf7a', (0, (2.4, 1.8))),
    ('lbscpi', 'LBS + CPI', '#eb6834', '-'),
)
LINE_Z = dict(lbs=3.5)
NO_BAND = ('lbs',)
FRONT_MARKER = dict(lbs=dict(ms=4.0, zorder=4.6))
RESTARTS = dict(label=r'CPI, best of $k$', color='#52514e', ls=':', lw=1.3)
TABLE_ROWS = (
    ('cpi2', r'CPI, $|\mathcal Z|=2$', 'CPI, \\|Z\\|=2'), ('cpi4', r'CPI, $|\mathcal Z|=4$', 'CPI, \\|Z\\|=4'),
    ('bestk', r'CPI, best of $k$', 'CPI, best of k'), ('tae', 'Trial-and-Error', 'Trial-and-Error'),
    ('lbs', 'LBS', 'LBS'), ('lbscpi', 'LBS + CPI', 'LBS + CPI'), ('npg', 'NPG-PD', 'NPG-PD'),
    ('npg_at_cpi', 'NPG-PD at the sweeps of CPI', 'NPG-PD at the sweeps of CPI'),
    ('cgcp', 'CGCP (belief space)', 'CGCP, belief space (no memory budget)'),
    ('cgcp_ub', 'Upper bound (belief space)', 'certified upper bound, belief space'))
MARKERS = dict(cpi2='o', cpi4='s', npg='P', lbs='^', lbscpi='D', cgcp='X')
CGCP_STYLE = dict(label='CGCP (belief space)', color='#0b0b0b', marker='X')
TITLE = r'RS$(5,{rocks})$, $c={c:g}$'
COMPACT = dict(curves=(3.2, 3.6), front=(2.53, 3.6))
FRONT_LABELS_COMPACT = {'evaluated policies': 'policies evaluated', 'best mixture of two found': 'best mixture found',
                        'best single policy found': 'best single found', 'CGCP (belief space)': 'CGCP'}
FRONT_LEGEND = 'side'
FRONT_SIZE = (5.0, 4.35)
FRONT_OMIT = ('evaluated policies', 'best single policy found')
FRONT_LABELS_SIDE = {'best mixture of two found': 'best mixture\nof two found', 'CGCP (belief space)': 'CGCP\n(belief space)'}
MAZE_START_LABEL = dict(copycoin='copy + coin start', p1='from CPI\'s (P1) policy')
FONT = dict(family='DejaVu Sans', size=8, title=8.5, label=8, ticks=7, legend=6.5)
COLORS = dict(text='#0b0b0b', muted='#52514e', axes='#8a8984', grid='#e4e3df', cloud='#8a8984')
SHOW_NPG = False
CLOUD_NPG = False
MIN_SHARE = 0.5
FEAS = 1e-9

warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom')


def style():
    plt.rcParams.update({
        'font.family': FONT['family'], 'font.size': FONT['size'], 'axes.titlesize': FONT['title'],
        'axes.labelsize': FONT['label'], 'xtick.labelsize': FONT['ticks'], 'ytick.labelsize': FONT['ticks'],
        'axes.edgecolor': COLORS['axes'], 'axes.labelcolor': COLORS['text'], 'xtick.color': COLORS['muted'],
        'ytick.color': COLORS['muted'], 'text.color': COLORS['text'], 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.grid': True, 'grid.color': COLORS['grid'], 'grid.linewidth': 0.6,
        'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def save(fig, out, name):
    for ext, kw in (('pdf', {}), ('png', dict(dpi=250))):
        path = out / f'{name}.{ext}'
        try:
            fig.savefig(path, **kw)
        except OSError:
            path = out / f'{name}_new.{ext}'
            fig.savefig(path, **kw)
    plt.close(fig)
    print('wrote', out / name)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def best_so_far(phis, rs, c, offset=0):
    best, out = -np.inf, []
    for n, (p, r) in enumerate(zip(phis, rs)):
        if p is not None and r is not None and p >= c - FEAS:
            best = max(best, r)
        out.append((n + offset, best))
    return out


def load(root):
    root = Path(root)
    D = dict(rock=defaultdict(lambda: defaultdict(dict)), npg=defaultdict(dict), tune=defaultdict(dict), cgcp={},
             maze=defaultdict(lambda: defaultdict(dict)), points=defaultdict(list), points_npg=defaultdict(list), tuned={})
    n = 0
    for f in sorted((root / 'rock').glob('*/result.json')):
        res, lg = _read(f), _read(f.with_name('log.json')) or {}
        job = res['job']
        k = int(job['task'][10])
        for rec in res['records']:
            c = round(float(rec['c']), 6)
            if job['kind'] == 'cpi':
                arm = 'tae' if job.get('p2_step') == 'naive' else {2: 'cpi2', 4: 'cpi4'}.get(job['Z'])
                sw = lg.get(str(rec['level'])) if isinstance(lg, dict) else None
                d = dict(J_r=rec['J_r'] if rec.get('feasible') else None, sweeps=rec['sweeps'])
                if sw:
                    phis = [x.get('J_phi') for x in sw]
                    rs = [x.get('J_r') for x in sw]
                    d.update(curve=best_so_far(phis, rs, c), phi=phis)
                    if job['Z'] == 2:
                        D['points'][k] += [(p, r) for p, r in zip(phis, rs) if p is not None and r is not None]
                D['rock'][(k, c)][arm][job['seed']] = d
            elif job['kind'] == 'bisect':
                if rec.get('J_r') is None:
                    continue
                cu = [(int(a), float(b)) for a, b in lg.get('curve', [])] if isinstance(lg, dict) else None
                D['rock'][(k, c)]['lbs'][job['seed']] = dict(J_r=rec['J_r'], sweeps=rec['sweeps'],
                                                            certificate=rec.get('certificate'), width=rec.get('width'),
                                                            curve=[x for x in cu if x[0] <= rec['sweeps']] if cu else None)
                D['rock'][(k, c)]['lbscpi'][job['seed']] = dict(
                    J_r=rec.get('polish_J_r', rec['J_r']), sweeps=rec['sweeps'] + rec.get('polish_sweeps', 0), curve=cu)
                if isinstance(lg, dict) and lg.get('points'):
                    D['points'][k] += [tuple(p) for p in lg['points']]
        n += 1
    for exp, key in (('p2_npg', 'npg'), ('p2_npg_tune', 'tune')):
        for f in sorted((root / exp).glob('*/result.json')):
            res = _read(f)
            rec, job = res['records'][0], res['job']
            c = round(float(job['c']), 6)
            if key == 'tune':
                D['tune'][(job['task'], c, job['start'])][(job['eta'], job['eta_lam'], job['seed'])] = rec
                continue
            lg = _read(f.with_name('log.json')) or {}
            d = dict(rec)
            if lg:
                d['curve'] = best_so_far(lg['J_phi'], lg['J_r'], c, rec.get('start_sweeps', 0))
                d['phi'] = lg['J_phi']
                if job['task'].startswith('rocksample'):
                    D['points_npg'][int(job['task'][10])] += list(zip(lg['J_phi'][1:], lg['J_r'][1:]))
            D['npg'][(job['task'], c, job['start'])][job['seed']] = d
    D['tuned'] = _read(root / 'p2_npg_tune' / 'tuned.json') or {}
    for f in sorted((root / 'p2_cgcp').glob('*/result.json')):
        res = _read(f)
        rec = res['records'][0]
        key = (res['job']['task'], round(float(res['job']['c']), 6))
        if key not in D['cgcp'] or res['job'].get('timeout', 0) > D['cgcp'][key].get('timeout', 0):
            D['cgcp'][key] = rec
    for f in sorted((root / 'tgap').glob('*/result.json')):
        res = _read(f)
        job = res['job']
        for rec in res['records']:
            c = round(float(rec['c']), 6)
            if job['kind'] == 'cpi' and job['Z'] in (2, 4):
                D['maze'][(job['task'], c)][f'cpi{job["Z"]}'][job['seed']] = rec['J_r'] if rec.get('feasible') else None
            elif job['kind'] == 'bisect':
                D['maze'][(job['task'], c)]['lbscpi'][job['seed']] = rec.get('polish_J_r', rec.get('J_r'))
    npg = (f'{sum(len(v) for v in D["npg"].values())} NPG-PD runs, {sum(len(v) for v in D["tune"].values())} tuning runs, '
           if SHOW_NPG else '')
    print(f'read {n} RockSample runs of ours, {npg}{len(D["maze"])} polar-maze settings, {len(D["cgcp"])} CGCP results')
    return D


def tcrit(n):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
    except ImportError:
        return 1.96


def paired(a, b):
    s = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    d = np.array([a[k] - b[k] for k in s], float)
    if len(d) == 0:
        return dict(mean=np.nan, ci=np.nan, n=0, wins=0, maxabs=np.nan)
    ci = tcrit(len(d)) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    return dict(mean=float(d.mean()), ci=float(ci), n=len(d), wins=int((d > 1e-6).sum()), maxabs=float(np.abs(d).max()))


def expected_best_of_k(vals, k):
    v = np.sort(np.asarray(vals, float))
    n = len(v)
    if k >= n:
        return float(v[-1])
    w = np.array([comb(i, k - 1) for i in range(n)], float) / comb(n, k)
    return float((w * v).sum())


def values(arm_runs):
    return {s: d.get('J_r') for s, d in arm_runs.items()}


def npg_values(runs, at=None, sweeps_of=None):
    out = {}
    for s, d in runs.items():
        if at is None:
            out[s] = d.get('best_J_r')
        elif sweeps_of.get(s) and d.get('curve'):
            b = [v for n, v in d['curve'] if n <= sweeps_of[s]]
            out[s] = b[-1] if b and np.isfinite(b[-1]) else None
    return out


def rock_npg(D, k, c):
    return D['npg'].get((f'rocksample{k}_tight', c, 'dirichlet'), {})


def rock_table(D, out):
    cfgs = sorted(D['rock'])
    if not cfgs:
        return {}
    stats = {}
    for cfg in cfgs:
        g = D['rock'][cfg]
        base = values(g['cpi2'])
        npg = rock_npg(D, *cfg)
        cpi_sw = {s: d['sweeps'] for s, d in g['cpi2'].items()}
        row = dict(cpi2=dict(mean=float(np.mean([v for v in base.values() if v is not None])), n=len(base)))
        for arm in ('cpi4', 'tae', 'lbs', 'lbscpi'):
            row[arm] = paired(values(g[arm]), base)
        finals = [v for v in base.values() if v is not None]
        k = int(round(np.median([d['sweeps'] for d in g['lbs'].values()]) / np.median(list(cpi_sw.values())))) \
            if g['lbs'] else 1
        row['bestk'] = dict(mean=expected_best_of_k(finals, k) - np.mean(finals), ci=np.nan, n=len(finals), k=k)
        cg = D['cgcp'].get((f'rocksample{cfg[0]}_tight', cfg[1]))
        mean_base = float(np.mean(finals))
        row['cgcp'] = dict(mean=cg['J_r'] - mean_base, ci=np.nan, n=1, abs=cg['J_r']) if cg and cg.get('J_r') is not None \
            else dict(mean=np.nan, ci=np.nan, n=0)
        row['cgcp_ub'] = dict(mean=cg['upper_bound'] - mean_base, ci=np.nan, n=1, abs=cg['upper_bound']) \
            if cg and np.isfinite(cg.get('upper_bound', np.nan)) else dict(mean=np.nan, ci=np.nan, n=0)
        row['npg'] = paired(npg_values(npg), base)
        row['npg_at_cpi'] = paired(npg_values(npg, at=True, sweeps_of=cpi_sw), base)
        row['npg_vs_lbscpi'] = paired(npg_values(npg), values(g['lbscpi']))
        row['npg_never'] = sum(d.get('best_J_r') is None for d in npg.values())
        row['npg_n'] = len(npg)
        row['npg_last_infeasible'] = float(np.mean([not d['last_feasible'] for d in npg.values()])) if npg else np.nan
        row['npg_tail_infeasible'] = float(np.median([d['infeasible_share_tail'] for d in npg.values()])) if npg else np.nan
        t = D['tuned'].get(f'rocksample{cfg[0]}_tight|{cfg[1]:g}|dirichlet')
        row['tuned'] = (t['eta'], t['eta_lam']) if t else None
        row['cert'] = max([d['certificate'] for d in g['lbs'].values() if d.get('certificate') is not None] or [np.nan])
        row['sweeps'] = dict(cpi2=float(np.median(list(cpi_sw.values()))),
                             lbscpi=float(np.median([d['sweeps'] for d in g['lbscpi'].values()])) if g['lbscpi'] else np.nan,
                             npg=float(np.median([d['sweeps'] + d.get('start_sweeps', 0) for d in npg.values()])) if npg else np.nan)
        stats[cfg] = row
    head = '| Method | ' + ' | '.join(f'{k} rocks, c={c:g}' for k, c in cfgs) + ' |'
    md = ['# (P2) on RockSample', '', head, '|---|' + '---|' * len(cfgs)]
    rows_shown = [x for x in TABLE_ROWS if SHOW_NPG or not x[0].startswith('npg')]
    for key, _, label in rows_shown:
        cells = []
        for cfg in cfgs:
            r = stats[cfg][key]
            if key == 'cpi2':
                cells.append(f"{r['mean']:.2f} (n={r['n']})")
            elif key == 'bestk':
                cells.append(f"{r['mean']:+.2f} (k={r['k']})")
            elif key in ('cgcp', 'cgcp_ub'):
                cells.append('—' if r['n'] == 0 else f"{r['mean']:+.2f} ({r['abs']:.2f})")
            elif r['n'] == 0:
                cells.append('—')
            else:
                cells.append(f"{r['mean']:+.2f} ± {r['ci']:.2f} (n={r['n']})")
        md.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    extra = [('Largest certificate', lambda r: f"{r['cert']:.2e}"),
             ('NPG-PD never feasible', lambda r: f"{r['npg_never']}/{r['npg_n']}"),
             ('NPG-PD last iterate infeasible', lambda r: f"{r['npg_last_infeasible']:.2f}"),
             ('NPG-PD infeasible share, second half (median)', lambda r: f"{r['npg_tail_infeasible']:.2f}"),
             ('NPG-PD (eta, eta_lam)', lambda r: f"{r['tuned']}" if r['tuned'] else '—'),
             ('Median sweeps CPI / LBS + CPI / NPG-PD', lambda r: f"{r['sweeps']['cpi2']:.0f} / {r['sweeps']['lbscpi']:.0f} / {r['sweeps']['npg']:.0f}"),
             ('NPG-PD minus LBS + CPI (max abs)', lambda r: f"{r['npg_vs_lbscpi']['mean']:+.3f} ± {r['npg_vs_lbscpi']['ci']:.3f} ({r['npg_vs_lbscpi']['maxabs']:.3f})")]
    if not SHOW_NPG:
        extra = [(label, fn) for label, fn in extra if 'NPG' not in label]
        extra[-1] = ('Median sweeps CPI / LBS + CPI', lambda r: f"{r['sweeps']['cpi2']:.0f} / {r['sweeps']['lbscpi']:.0f}")
    for label, fn in extra:
        md.append(f'| {label} | ' + ' | '.join(fn(stats[cfg]) for cfg in cfgs) + ' |')
    md += ['', 'Rows after the first: mean paired difference from CPI (|Z| = 2) over the seeds both have, with the 95% '
           'half-width (CGCP and its bound: one deterministic run, minus the mean of CPI, absolute value in brackets). '
           + ('NPG-PD is its best feasible iterate within its budget, or within as many sweeps as CPI used on '
              'that seed; seeds where it is never feasible are left out of the pair and counted separately. ' if SHOW_NPG else '')
           + 'Best of k is the expected best of k CPI runs, with k the ratio of the median sweeps of LBS and CPI.']
    (out / 'table_p2_rock.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    ks = sorted({k for k, _ in cfgs})
    tex = [r'\begin{tabular}{l' + 'c' * len(cfgs) + '}', r'\toprule',
           '& ' + ' & '.join(rf'\multicolumn{{{sum(1 for kk, _ in cfgs if kk == k)}}}{{c}}{{{k} rocks}}' for k in ks) + r'\\']
    col = 2
    rules = []
    for k in ks:
        m = sum(1 for kk, _ in cfgs if kk == k)
        rules.append(rf'\cmidrule(lr){{{col}-{col + m - 1}}}')
        col += m
    tex += [''.join(rules), '$c$ & ' + ' & '.join(f'{c:g}' for _, c in cfgs) + r'\\', r'\midrule']
    for key, label, _ in rows_shown:
        cells = []
        for cfg in cfgs:
            r = stats[cfg][key]
            cells.append(f"{r['mean']:.2f}" if key == 'cpi2' else ('—' if r.get('n', 1) == 0 else f"${r['mean']:+.2f}$"))
        tex.append(f'{label} & ' + ' & '.join(cells) + r'\\')
    tex += [r'\midrule', r'Largest certificate ($\times10^{-3}$) & ' + ' & '.join(f"{stats[cfg]['cert'] * 1e3:.1f}" for cfg in cfgs) + r'\\',
            r'\bottomrule', r'\end{tabular}']
    (out / 'table_p2_rock.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print('wrote', out / 'table_p2_rock')
    return stats


def maze_table(D, root, out):
    if not D['maze'] and not D['npg']:
        return {}
    try:
        import tmaze_gap as tg
        from report_paper import v2_found
        found = v2_found(root)
    except Exception:
        tg, found = None, {}
    keys = sorted({(t, c) for (t, c, st) in D['npg'] if t.startswith('tgap')} | set(D['maze']))
    rows, stats = [], {}
    for task, c in keys:
        K, s = int(task.split('_K')[1].split('_')[0]), int(task.split('_s')[1])
        v2 = tg.v2(c, K, s) if tg else np.nan
        dag = ''
        if not np.isfinite(v2) and (K, s, c) in found:
            v2, dag = found[(K, s, c)], ' †'
        vb = tg.v2_bar(c, K, s) if tg else np.nan
        g = D['maze'].get((task, c), {})
        med = lambda d: float(np.median([v for v in d.values() if v is not None])) if d and any(v is not None for v in d.values()) else np.nan
        cg = D['cgcp'].get((task, c), {})
        st = dict(K=K, s=s, c=c, v2=v2, dag=dag, vb=vb, cpi2=med(g.get('cpi2', {})), lbscpi=med(g.get('lbscpi', {})),
                  cgcp=cg.get('J_r', np.nan) if cg.get('J_r') is not None else np.nan, cgcp_ub=cg.get('upper_bound', np.nan))
        for start in ('copycoin', 'p1'):
            runs = D['npg'].get((task, c, start), {})
            best = [d['best_J_r'] for d in runs.values() if d.get('best_J_r') is not None]
            st[start] = dict(n=len(runs), feas=len(best), med=float(np.median(best)) if best else np.nan,
                             maxphi=float(np.median([d['max_J_phi'] for d in runs.values()])) if runs else np.nan)
        stats[(task, c)] = st
    fmt = lambda x: '—' if not np.isfinite(x) else (f'{x:.2e}' if 0 < abs(x) < 1e-3 else f'{x:.4f}')
    if not SHOW_NPG:
        md = ['# (P2) on the polar maze', '', '| K | s | c | v2 | v2_bar | CPI \\|Z\\|=2 | LBS + CPI | CGCP, belief space [upper bound] |',
              '|---|---|---|---|---|---|---|---|---|']
        for (task, c), st in stats.items():
            md.append(f"| {st['K']} | {st['s']} | {c:g} | {fmt(st['v2'])}{st['dag']} | {fmt(st['vb'])} | {fmt(st['cpi2'])} | "
                      f"{fmt(st['lbscpi'])} | {fmt(st['cgcp'])} [{fmt(st['cgcp_ub'])}] |")
        md += ['', 'Medians over seeds. † No closed form for v2 here: the best policy found, a lower bound.']
        (out / 'table_p2_maze.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
        tex = [r'\begin{tabular}{rrcccccc}', r'\toprule',
               r'$K$ & $s$ & $c$ & $v_2$ & $\bar v_2$ & CPI & LBS + CPI & CGCP \\', r'\midrule']
        for (task, c), st in stats.items():
            tex.append(f"{st['K']} & {st['s']} & {c:g} & {fmt(st['v2'])}{'$^\\dagger$' if st['dag'] else ''} & {fmt(st['vb'])} & "
                       f"{fmt(st['cpi2'])} & {fmt(st['lbscpi'])} & {fmt(st['cgcp'])} \\\\")
        tex += [r'\bottomrule', r'\end{tabular}']
        (out / 'table_p2_maze.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
        print('wrote', out / 'table_p2_maze')
        return stats
    md = ['# (P2) on the survey maze with NPG-PD', '',
          '| K | s | c | v2 | v2_bar | CPI |Z|=2 | LBS + CPI | NPG-PD, copy + coin: feasible, median best [median max J_phi] | '
          'NPG-PD from CPI (P1) policy: feasible, median best | CGCP, belief space [upper bound] |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for (task, c), st in stats.items():
        a, b = st['copycoin'], st['p1']
        ca = '—' if a['n'] == 0 else f"{a['feas']}/{a['n']}, {fmt(a['med'])} [{fmt(a['maxphi'])}]"
        cb = '—' if b['n'] == 0 else f"{b['feas']}/{b['n']}, {fmt(b['med'])}"
        md.append(f"| {st['K']} | {st['s']} | {c:g} | {fmt(st['v2'])}{st['dag']} | {fmt(st['vb'])} | {fmt(st['cpi2'])} | "
                  f"{fmt(st['lbscpi'])} | {ca} | {cb} | {fmt(st['cgcp'])} [{fmt(st['cgcp_ub'])}] |")
    md += ['', 'Medians over seeds. † No closed form for v2 here: the best policy found, a lower bound. NPG-PD from '
           'copy + coin starts where our arms start; from the (P1) policy it starts feasible, storing the cue.']
    (out / 'table_p2_maze.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    tex = [r'\begin{tabular}{rrccccccc}', r'\toprule',
           r'$K$ & $s$ & $v_2$ & $\bar v_2$ & CPI & LBS + CPI & NPG-PD & NPG-PD from (P1) & CGCP \\', r'\midrule']
    for (task, c), st in stats.items():
        if abs(c - 0.75) > 1e-9:
            continue
        a, b = st['copycoin'], st['p1']
        cell = lambda x: '—' if x['n'] == 0 else 'infeasible' if x['feas'] == 0 else (
            fmt(x['med']) + ('' if x['feas'] == x['n'] else f" ({x['feas']}/{x['n']})"))
        tex.append(f"{st['K']} & {st['s']} & {fmt(st['v2'])}{'$^\\dagger$' if st['dag'] else ''} & {fmt(st['vb'])} & "
                   f"{fmt(st['cpi2'])} & {fmt(st['lbscpi'])} & {cell(a)} & {cell(b)} & {fmt(st['cgcp'])} \\\\")
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'table_p2_maze.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    print('wrote', out / 'table_p2_maze')
    return stats


def on_grid(curve, xs):
    n = np.array([a for a, _ in curve], float)
    v = np.array([b for _, b in curve], float)
    idx = np.searchsorted(n, xs, side='right') - 1
    o = np.full(len(xs), np.nan)
    o[idx >= 0] = v[idx[idx >= 0]]
    o[~np.isfinite(o)] = np.nan
    return o


def curve_panel(ax, D, cfg, view, xs):
    g = D['rock'][cfg]
    runs = dict(cpi2=g['cpi2'], cpi4=g['cpi4'], lbs=g['lbs'], lbscpi=g['lbscpi'], npg=rock_npg(D, *cfg))
    base_vals = [d['J_r'] for d in g['cpi2'].values() if d.get('J_r') is not None]
    base = float(np.median(base_vals)) if (view == 'relative' and base_vals) else 0.0
    for key, label, col, ls in SERIES:
        curves = [d['curve'] for d in runs[key].values() if d.get('curve')]
        if not curves:
            continue
        M = np.array([on_grid(cu, xs) for cu in curves]) - base
        ok = np.mean(np.isfinite(M), 0) >= MIN_SHARE
        med = np.nanmedian(M, 0)
        if key not in NO_BAND:
            ax.fill_between(xs[ok], np.nanpercentile(M, 25, 0)[ok], np.nanpercentile(M, 75, 0)[ok], color=col, alpha=0.13, lw=0)
        z = LINE_Z.get(key, 3)
        ax.plot(xs[ok], med[ok], color=col, ls=ls, lw=1.4, label=label, zorder=z)
        stop = float(np.median([cu[-1][0] for cu in curves]))
        j = min(np.searchsorted(xs, stop, side='right') - 1, len(xs) - 1)
        if j >= 0 and ok[j]:
            ax.plot([stop], [med[j]], 'o', ms=4, color=col, mec='white', mew=0.7, zorder=z + 1)
    if base_vals:
        m = float(np.median([d['sweeps'] for d in g['cpi2'].values()]))
        ks = np.arange(1, len(base_vals) + 1)
        bx, by = ks * m, np.array([expected_best_of_k(base_vals, k) for k in ks]) - base
        keep = bx <= xs[-1]
        ax.plot(bx[keep], by[keep], color=RESTARTS['color'], ls=RESTARTS['ls'], lw=RESTARTS['lw'], label=RESTARTS['label'], zorder=2)
    if view == 'relative':
        ax.axhline(0, color=COLORS['muted'], lw=0.7, zorder=0)
        ax.set_ylim(-1.0, None)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(2, xs[-1] * 1.05)


def fig_curves(D, out, cfgs, name, compact=False):
    if not cfgs:
        return
    longest = max(cu[-1][0] for cfg in cfgs for runs in (D['rock'][cfg]['cpi2'], D['rock'][cfg]['lbscpi']) + ((rock_npg(D, *cfg),) if SHOW_NPG else ())
                  for d in runs.values() for cu in [d.get('curve')] if cu)
    xs = np.unique(np.round(np.logspace(0, np.log10(longest), 260))).astype(float)
    size = COMPACT['curves'] if compact else (2.9 * len(cfgs) + 0.3, 3.9)
    fig, axs = plt.subplots(2, len(cfgs), figsize=size, sharex=True, squeeze=False)
    for j, cfg in enumerate(cfgs):
        for i, view in enumerate(('absolute', 'relative')):
            curve_panel(axs[i, j], D, cfg, view, xs)
        axs[0, j].set_title(TITLE.format(rocks=cfg[0], c=cfg[1]), loc='left')
        axs[1, j].set_xlabel('$N$')
    axs[0, 0].set_ylabel('$J_r$')
    axs[1, 0].set_ylabel(r'$\Delta J_r$')
    h, l = axs[0, 0].get_legend_handles_labels()
    if compact:
        rows = int(np.ceil(len(h) / 2))
        fig.tight_layout(w_pad=0.5, h_pad=0.5, rect=(0, (0.12 * rows + 0.04) / size[1], 1, 1))
        fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=FONT['legend'], handlelength=1.8,
                   columnspacing=1.2, labelspacing=0.3)
    else:
        fig.tight_layout(w_pad=0.6, h_pad=0.6)
        axs[0, 0].legend(h, l, loc='lower right', fontsize=FONT['legend'], frameon=True, framealpha=0.92,
                         edgecolor='none', handlelength=1.8, labelspacing=0.3, borderpad=0.3).set_zorder(10)
    save(fig, out, name)


def fig_feasibility(D, out):
    cfgs = sorted(D['rock'])
    if not cfgs or not any(rock_npg(D, *cfg) for cfg in cfgs):
        return
    ks = sorted({k for k, _ in cfgs})
    ncol = max(sum(1 for k, _ in cfgs if k == kk) for kk in ks)
    fig, axs = plt.subplots(len(ks), ncol, figsize=(2.3 * ncol + 0.3, 1.8 * len(ks) + 0.45), sharex=True, sharey=True,
                            squeeze=False)
    for i, k in enumerate(ks):
        for j, cfg in enumerate([cf for cf in cfgs if cf[0] == k]):
            ax = axs[i, j]
            c = cfg[1]
            for key, runs, col, label in (('cpi2', D['rock'][cfg]['cpi2'], '#2a78d6', r'CPI, $|\mathcal{Z}|=2$'),
                                          ('npg', rock_npg(D, *cfg), '#8e44ad', 'NPG-PD')):
                trajs = [d['phi'] for d in runs.values() if d.get('phi')]
                if not trajs:
                    continue
                L = max(len(t) for t in trajs)
                share = np.full(L, np.nan)
                for n in range(L):
                    once = [t for t in trajs if any(p is not None and p >= c - FEAS for p in t[:n + 1])]
                    if once:
                        cur = [t[min(n, len(t) - 1)] for t in once]
                        share[n] = np.mean([p is None or p < c - FEAS for p in cur])
                ax.plot(np.arange(L), share, color=col, lw=1.3, label=label)
            ax.set_title(TITLE.format(rocks=k, c=c), loc='left')
            ax.set_xscale('symlog', linthresh=10)
            ax.set_ylim(-0.03, 1.03)
        axs[i, 0].set_ylabel('infeasible share')
    for ax in axs[-1]:
        ax.set_xlabel('$N$')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=FONT['legend'])
    save(fig, out, 'fig_p2_feasibility')


def fig_tuning(D, out):
    settings = sorted(k for k in D['tune'] if k[0].startswith('rocksample'))
    if not settings:
        return
    ks = sorted({int(t[10]) for t, _, _ in settings})
    ncol = max(sum(1 for t, _, _ in settings if int(t[10]) == k) for k in ks)
    fig, axs = plt.subplots(len(ks), ncol, figsize=(2.2 * ncol + 0.6, 2.0 * len(ks) + 0.2), squeeze=False)
    for i, k in enumerate(ks):
        row = [s for s in settings if int(s[0][10]) == k]
        for j, (task, c, st) in enumerate(row):
            ax = axs[i, j]
            runs = D['tune'][(task, c, st)]
            etas = sorted({e for e, _, _ in runs})
            lams = sorted({l for _, l, _ in runs})
            M = np.full((len(etas), len(lams)), np.nan)
            for a_, e in enumerate(etas):
                for b_, l in enumerate(lams):
                    v = [r['best_J_r'] if r['best_J_r'] is not None else -np.inf for (ee, ll, s), r in runs.items()
                         if ee == e and ll == l]
                    M[a_, b_] = np.mean(v) if v and np.all(np.isfinite(v)) else np.nan
            rel = M - np.nanmax(M)
            cmap = plt.get_cmap('Purples_r').copy()
            cmap.set_bad(COLORS['grid'])
            im = ax.imshow(np.ma.masked_invalid(np.clip(rel, -1.0, 0)), cmap=cmap, vmin=-1.0, vmax=0, aspect='auto')
            for a_ in range(len(etas)):
                for b_ in range(len(lams)):
                    ax.text(b_, a_, 'infeas.' if not np.isfinite(rel[a_, b_]) else f'{rel[a_, b_]:.2f}', ha='center',
                            va='center', fontsize=5.5, color='white' if np.isfinite(rel[a_, b_]) and rel[a_, b_] < -0.5 else COLORS['text'])
            t = D['tuned'].get(f'{task}|{c:g}|{st}')
            if t and t['eta'] in etas and t['eta_lam'] in lams:
                ax.add_patch(plt.Rectangle((lams.index(t['eta_lam']) - 0.5, etas.index(t['eta']) - 0.5), 1, 1, fill=False,
                                           ec=COLORS['text'], lw=1.2))
            ax.set_xticks(range(len(lams)), [f'{l:g}' for l in lams])
            ax.set_yticks(range(len(etas)), [f'{e:g}' for e in etas])
            ax.grid(False)
            ax.set_title(TITLE.format(rocks=k, c=c), loc='left')
            if i == len(ks) - 1:
                ax.set_xlabel(r'$\eta_\lambda$')
            if j == 0:
                ax.set_ylabel(r'$\eta$')
        for ax in axs[i, len(row):]:
            ax.set_visible(False)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    cb.set_label(r'$J_r$ minus the best cell', fontsize=FONT['ticks'])
    save(fig, out, 'fig_p2_tuning')


def upper_hull(P):
    P = P[np.lexsort((-P[:, 1], P[:, 0]))]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    hull = []
    for p in P:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    H = np.array(hull)
    return H[np.r_[True, np.diff(H[:, 0]) > 0]]


def best_single(P, c):
    m = P[:, 0] >= c - FEAS
    return float(P[m, 1].max()) if m.any() else np.nan


def fig_front(D, out, compact=False):
    ks = sorted(k for k in D['points'] if D['points'][k])
    if not ks:
        return
    side = FRONT_LEGEND == 'side' and not compact
    size = COMPACT['front'] if compact else (FRONT_SIZE if len(ks) == 2 else (5.0, 1.9 * len(ks) + 0.55))
    fig, axs = plt.subplots(len(ks), 1, figsize=size, sharex=True, squeeze=False)
    rng = np.random.default_rng(0)
    gaps = {}
    for ax, k in zip(axs[:, 0], ks):
        P = np.array([p for p in D['points'][k] + (D['points_npg'][k] if CLOUD_NPG else [])
                      if p[0] is not None and p[1] is not None], float)
        sub = P if len(P) <= 40000 else P[rng.choice(len(P), 40000, replace=False)]
        ax.scatter(sub[:, 0], sub[:, 1], s=1.5, color=COLORS['cloud'], alpha=0.12, lw=0, rasterized=True, zorder=0,
                   label='evaluated policies')
        H = upper_hull(P)
        ax.plot(H[:, 0], H[:, 1], color=COLORS['text'], lw=0.9, zorder=2, label='best mixture of two found')
        cs = sorted(c for kk, c in D['rock'] if kk == k)
        singles = [best_single(P, c) for c in cs]
        ax.plot(cs, singles, ls='none', marker='_', ms=11, mew=1.4, color=COLORS['text'], zorder=3,
                label='best single policy found')
        ys = [np.interp(c, H[:, 0], H[:, 1]) for c in cs]
        for c, f in zip(cs, singles):
            ax.axvline(c, color=COLORS['cloud'], lw=0.7, ls=':', zorder=1)
            gaps[(k, c)] = float(np.interp(c, H[:, 0], H[:, 1]) - f)
        for key, label, col, _ in SERIES:
            pts = []
            for c in cs:
                runs = rock_npg(D, k, c) if key == 'npg' else D['rock'][(k, c)][key]
                if key == 'npg':
                    ends = [(d['best_J_phi'], d['best_J_r']) for d in runs.values() if d.get('best_J_r') is not None]
                else:
                    ends = [(c, d['J_r']) for d in runs.values() if d.get('J_r') is not None]
                if ends:
                    pts.append(np.median(np.array(ends), 0))
            if pts:
                pts = np.array(pts)
                kw = dict(ms=5, zorder=4, mew=0.6) | FRONT_MARKER.get(key, {})
                ax.plot(pts[:, 0], pts[:, 1], ls='none', marker=MARKERS[key], color=col, mec='white', label=label, **kw)
                ys += list(pts[:, 1])
        cg = [(rec['J_phi'], rec['J_r']) for (task, c), rec in D['cgcp'].items()
              if task == f'rocksample{k}_tight' and rec.get('J_r') is not None]
        if cg:
            cg = np.array(cg)
            ax.plot(cg[:, 0], cg[:, 1], ls='none', marker=CGCP_STYLE['marker'], ms=5.5, color=CGCP_STYLE['color'],
                    mec='white', mew=0.6, zorder=5, label=CGCP_STYLE['label'])
            ys += list(cg[:, 1])
        ax.set_ylim(min(ys) - 0.6, max(H[:, 1].max(), max(ys)) + 0.25)
        ax.set_ylabel('$J_r$')
        ax.text(0.98, 0.94, f'RS$(5,{k})$', transform=ax.transAxes, ha='right', va='top', fontsize=FONT['title'])
    axs[-1, 0].set_xlabel(r'$J_\varphi$')
    axs[-1, 0].set_xlim(-0.02, max(np.array(D['points'][k], float)[:, 0].max() for k in ks) + 0.02)
    h, l = axs[0, 0].get_legend_handles_labels()
    for hd in h:
        if hd.get_label() == 'evaluated policies':
            hd.set_alpha(0.6)
    if side:
        keep = [i for i, x in enumerate(l) if x not in FRONT_OMIT]
        h, l = [h[i] for i in keep], [FRONT_LABELS_SIDE.get(l[i], l[i]) for i in keep]
        fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 1))
        leg = fig.legend(h, l, loc='center right', ncol=1, frameon=False, fontsize=FONT['legend'], handlelength=1.6,
                         labelspacing=0.9, borderaxespad=0.0, handletextpad=0.6)
        fig.canvas.draw()
        w = leg.get_window_extent().width / fig.dpi / size[0]
        fig.tight_layout(h_pad=0.4, rect=(0, 0, 1 - w - 0.015, 1))
        save(fig, out, 'fig_p2_front')
        return gaps
    ncol = 2 if compact else 3
    if compact:
        l = [FRONT_LABELS_COMPACT.get(x, x) for x in l]
    rows = int(np.ceil(len(h) / ncol))
    fig.tight_layout(h_pad=0.4, rect=(0, (0.115 * rows + 0.02) / size[1], 1, 1))
    fig.legend(h, l, loc='lower center', ncol=ncol, frameon=False, fontsize=FONT['legend'],
               columnspacing=0.9 if compact else 1.2, handlelength=1.5 if compact else 2.0)
    save(fig, out, 'fig_p2_front_compact' if compact else 'fig_p2_front')
    return gaps


def numbers(rock, maze, gaps, out):
    L = ['# In-text numbers for (P2) (generated by plot_p2.py)', '', '## RockSample', '']
    for cfg, r in sorted(rock.items()):
        k, c = cfg
        v, a, b = r['npg_vs_lbscpi'], r['npg'], r['npg_at_cpi']
        if not SHOW_NPG:
            L.append(f"- {k} rocks, c = {c:g}: LBS minus CPI {r['lbs']['mean']:+.3f}, LBS + CPI minus CPI {r['lbscpi']['mean']:+.3f}; "
                     f"largest certificate {r['cert']:.1e}."
                     + (f" Best mixture of two found minus best single found: {gaps[cfg]:.3f}." if cfg in gaps else '')
                     + (f" CGCP in belief space (no memory budget) {r['cgcp']['abs']:.3f}, certified upper bound "
                        f"{r['cgcp_ub']['abs']:.3f} on every policy and mixture; LBS + CPI mean "
                        f"{r['cpi2']['mean'] + r['lbscpi']['mean']:.3f}." if r['cgcp']['n'] else ''))
            continue
        L.append(f"- {k} rocks, c = {c:g}: NPG-PD (tuned {r['tuned']}, {r['sweeps']['npg']:.0f} sweeps) minus LBS + CPI "
                 f"{v['mean']:+.3f} ± {v['ci']:.3f} (largest |difference| {v['maxabs']:.3f}, n = {v['n']}); minus CPI "
                 f"{a['mean']:+.3f} ± {a['ci']:.3f} (higher on {a['wins']}/{a['n']}); at CPI's sweeps "
                 f"({r['sweeps']['cpi2']:.0f}) minus CPI {b['mean']:+.3f} ± {b['ci']:.3f} (higher on {b['wins']}/{b['n']}). "
                 f"Never feasible {r['npg_never']}/{r['npg_n']}; last iterate infeasible in {r['npg_last_infeasible']:.0%} of "
                 f"runs; median share of infeasible iterates in the second half {r['npg_tail_infeasible']:.0%}. "
                 f"LBS + CPI minus CPI {r['lbscpi']['mean']:+.3f}; largest certificate {r['cert']:.1e}."
                 + (f" Best mixture of two found minus best single found: {gaps[cfg]:.3f}." if cfg in gaps else '')
                 + (f" CGCP in belief space (no memory budget) {r['cgcp']['abs']:.3f}, certified upper bound "
                    f"{r['cgcp_ub']['abs']:.3f} on every policy and mixture; LBS + CPI mean "
                    f"{r['cpi2']['mean'] + r['lbscpi']['mean']:.3f}." if r['cgcp']['n'] else ''))
    if maze:
        L += ['', '## Polar maze (survey maze in the code)', '']
        for (task, c), st in maze.items():
            a, b = st['copycoin'], st['p1']
            if not SHOW_NPG:
                L.append(f"- K = {st['K']}, s = {st['s']}, c = {c:g}: v2 = {st['v2']:.3g}{st['dag']}, v2_bar = {st['vb']:.3g}; "
                         f"CPI {st['cpi2']:.3g}, LBS + CPI {st['lbscpi']:.3g}."
                         + (f" CGCP (belief space) {st['cgcp']:.3g}, upper bound {st['cgcp_ub']:.3g}." if np.isfinite(st['cgcp']) else ''))
                continue
            if a['n'] == 0 and b['n'] == 0:
                continue
            L.append(f"- K = {st['K']}, s = {st['s']}, c = {c:g}: v2 = {st['v2']:.3g}{st['dag']}, v2_bar = {st['vb']:.3g}; "
                     f"CPI {st['cpi2']:.3g}, LBS + CPI {st['lbscpi']:.3g}; NPG-PD from copy + coin feasible in "
                     f"{a['feas']}/{a['n']} (median best {a['med']:.3g}, median largest J_phi {a['maxphi']:.3g}); from the (P1) "
                     f"policy feasible in {b['feas']}/{b['n']} (median best {b['med']:.3g})."
                     + (f" CGCP (belief space) {st['cgcp']:.3g}, upper bound {st['cgcp_ub']:.3g}." if np.isfinite(st['cgcp']) else ''))
    (out / 'numbers_p2.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('wrote', out / 'numbers_p2.md')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='runs_paper')
    ap.add_argument('--out', default='reports_p2')
    ap.add_argument('--only', default='', help='comma list of: table, maze, curves, feasibility, tuning, front, numbers')
    ap.add_argument('--all-curves', action='store_true', help='also the learning curves at every threshold')
    ap.add_argument('--npg', action='store_true', help='also show the NPG-PD baseline of an earlier draft (needs p2_npg/)')
    ap.add_argument('--no-npg', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--cloud-ours-only', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--compact', action='store_true',
                    help='also half-width versions of the curves (3.2 x 3.6 in) and the front (2.53 x 3.6 in), for the '
                         'side-by-side float: fig_p2_curves_compact and fig_p2_front_compact')
    a = ap.parse_args(argv)
    global SERIES, SHOW_NPG, CLOUD_NPG
    if a.npg:
        SHOW_NPG = CLOUD_NPG = True
    if not SHOW_NPG:
        SERIES = tuple(x for x in SERIES if x[0] != 'npg')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    want = lambda x: not a.only or x in a.only.split(',')
    style()
    D = load(a.root)
    rock = rock_table(D, out) if (want('table') or want('numbers')) else {}
    maze = maze_table(D, a.root, out) if (want('maze') or want('numbers')) else {}
    tight = [max((cf for cf in D['rock'] if cf[0] == k), key=lambda cf: cf[1]) for k in sorted({k for k, _ in D['rock']})]
    if want('curves'):
        fig_curves(D, out, tight, 'fig_p2_curves')
        if a.all_curves:
            fig_curves(D, out, sorted(D['rock']), 'fig_p2_curves_all')
        if a.compact:
            fig_curves(D, out, tight, 'fig_p2_curves_compact', compact=True)
    if want('feasibility') and SHOW_NPG:
        fig_feasibility(D, out)
    if want('tuning') and SHOW_NPG:
        fig_tuning(D, out)
    gaps = fig_front(D, out) if (want('front') or want('numbers')) else {}
    if a.compact and want('front'):
        fig_front(D, out, compact=True)
    if want('numbers'):
        numbers(rock, maze, gaps or {}, out)


if __name__ == '__main__':
    main()
