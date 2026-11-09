"""Tables, figures and in-text numbers from the run logs."""

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import chain
warnings.filterwarnings('ignore', message='All-NaN slice encountered')
warnings.filterwarnings('ignore', message='Mean of empty slice')
import tmaze_gap as tg

INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
FEAS = 1e-9


def style():
    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2,
        'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
        'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
        'legend.frameon': False, 'figure.dpi': 150, 'savefig.bbox': 'tight', 'pdf.fonttype': 42})


def save(fig, out, name):
    for ext, kw in (('pdf', {}), ('png', dict(dpi=200))):
        path = out / f'{name}.{ext}'
        try:
            fig.savefig(path, **kw)
        except OSError:
            alt = out / f'{name}_new.{ext}'
            fig.savefig(alt, **kw)
            print(f'could not overwrite {path} (open in a viewer?); wrote {alt} instead')
    plt.close(fig)


V2_FOUND = {(4, 1, 0.75): 0.018871026589, (4, 2, 0.75): 0.248045384062}


def v2_found(root):
    f = Path(root) / 'tgap_search' / 'results.json'
    if not f.exists():
        return dict(V2_FOUND)
    return {(r['K'], r['s'], r['c']): r['best'] for r in json.loads(f.read_text(encoding='utf-8'))}

_LOADED = {}


def load(root, exp):
    key = (str(Path(root).resolve()), exp)
    if key not in _LOADED:
        n = len(list((Path(root) / exp).glob('*/result.json')))
        if n:
            print(f'reading {Path(root) / exp} ({n} runs) ...', flush=True)
        _LOADED[key] = _load(root, exp)
    return _LOADED[key]


def _load(root, exp):
    rows = []
    for f in sorted((Path(root) / exp).glob('*/result.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        lg_path = f.parent / 'log.json'
        lg = json.loads(lg_path.read_text(encoding='utf-8')) if lg_path.exists() else {}
        for r in d['records']:
            log = lg.get(r['level'], lg) if d['job']['kind'] == 'cpi' else lg
            rows.append(dict(job=d['job'], rec=r, log=log, secs=d.get('secs')))
    return rows


def arm(row):
    j = row['job']
    if j['kind'] == 'bisect':
        return 'lagrangian'
    if j['kind'] == 'eg':
        return 'eg'
    if j['kind'] == 'cloud':
        return 'cloud'
    name = 'naive' if j.get('p2_step') == 'naive' else 'joint'
    if j.get('p2_from') == 'converged':
        name += '_conv'
    return f"{name}_Z{j['Z']}"


def threshold(row):
    r = row['rec']
    return float(r['c']) if r.get('c') is not None else float(row['job']['c'])


def by(rows, *keys):
    g = defaultdict(list)
    for r in rows:
        g[tuple(k(r) for k in keys)].append(r)
    return g


def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(x, p)) if len(x) else np.nan


def mean_ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    h = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return float(x.mean()), float(h), len(x)


def final_r(row):
    r = row['rec']
    return r['J_r'] if r.get('J_phi') is not None and r['J_phi'] >= threshold(row) - FEAS else np.nan


def incumbent_curve(row):
    c = threshold(row)
    if row['job']['kind'] == 'cpi':
        log = row['log']
        if isinstance(log, dict):
            log = log.get(str(row['rec'].get('level')), [])
        best, out = np.nan, []
        for k, l in enumerate(log[1:], start=1):
            if l['J_phi'] >= c - FEAS:
                best = l['J_r'] if not np.isfinite(best) else max(best, l['J_r'])
            out.append((k, best))
        return out
    return [(int(s), v if v is not None else np.nan) for s, v in row['log'].get('curve', [])]


def curve_band(rows, xs):
    M = np.full((len(rows), len(xs)), np.nan)
    for i, r in enumerate(rows):
        cv = incumbent_curve(r)
        if not cv:
            continue
        s = np.array([a for a, _ in cv], float)
        v = np.array([b for _, b in cv], float)
        idx = np.searchsorted(s, xs, side='right') - 1
        ok = idx >= 0
        M[i, ok] = v[idx[ok]]
    with np.errstate(all='ignore'):
        return (np.nanmedian(M, 0), np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0),
                np.mean(np.isfinite(M), 0))


def e1(root, out, notes):
    rows = load(root, 'chain')
    if not rows:
        return
    T = lambda r: r['rec'].get('decisions', r['job']['task'][5:])
    g = by(rows, lambda r: int(T(r)), threshold, arm)
    Ts = sorted({k[0] for k in g})
    cs = sorted({k[1] for k in g})

    def panel(ax, c, legend=True):
        floor = 1e-12
        ax.plot(Ts, [chain.v2(t, c) for t in Ts], color=INK2, ls='--', lw=1.0, label=r'$v_2$ (exact)')
        ax.plot(Ts, [chain.v2_bar(c)] * len(Ts), color=INK2, ls=':', lw=1.0, label=r'$\bar v_2$ (exact)')
        for name, col, mk, lab in (('joint_Z1', BLUE, 'o', 'joint greedy, $|Z|=1$'),
                                   ('joint_Z2', ORANGE, 's', 'joint greedy, spare bit'),
                                   ('lagrangian', AQUA, 'D', 'Lagrangian + bit')):
            med = [max(floor, q([final_r(r) for r in g.get((t, c, name), [])], 50)) for t in Ts]
            lo = [max(floor, q([final_r(r) for r in g.get((t, c, name), [])], 25)) for t in Ts]
            hi = [max(floor, q([final_r(r) for r in g.get((t, c, name), [])], 75)) for t in Ts]
            ax.plot(Ts, med, color=col, marker=mk, ms=4, lw=1.2, label=lab, mec='white', mew=0.8)
            ax.fill_between(Ts, lo, hi, color=col, alpha=0.12, lw=0)
        ax.set_yscale('log')
        ax.set_ylim(floor / 3, 2)
        ax.set_xticks([t for t in Ts if t % 2 == 0])
        ax.set_xlabel('decisions $T$')
        ax.set_ylabel(r'$J_r$ at $J_\varphi \geq c$')
        ax.set_title(f'Example 1, $c = {c:g}$', loc='left')
        if legend:
            ax.legend(loc='lower left')

    stats = []
    for c in cs:
        for t in Ts:
            jg2 = g.get((t, c, 'joint_Z2'), [])
            reach = sum(1 for r in jg2 if final_r(r) >= chain.v2_bar(c) - 1e-6)
            conv = [final_r(r) for r in g.get((t, c, 'joint_conv_Z2'), [])]
            lag = [final_r(r) for r in g.get((t, c, 'lagrangian'), [])]
            jg1 = [final_r(r) for r in g.get((t, c, 'joint_Z1'), [])]
            stats.append(dict(T=t, c=c, n=len(jg2), spare_reach=reach,
                              conv_max=max(conv) if conv else np.nan,
                              lag_err=max(abs(v - chain.v2_bar(c)) for v in lag) if lag else np.nan,
                              jg1_over=max(v - chain.v2(t, c) for v in jg1) if jg1 else np.nan,
                              jg1_med=q(jg1, 50), v2=chain.v2(t, c)))
    _table(out / 'tableA_chain.md', ['T', 'c', 'v2', 'jg1_med', 'jg1_over', 'spare_reach', 'n', 'conv_max', 'lag_err'], stats)
    fig, axs = plt.subplots(1, len(cs), figsize=(2.3 * len(cs), 2.5), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), cs):
        panel(ax, c, legend=False)
        ax.set_title(f'$c = {c:g}$', loc='left')
    for ax in np.atleast_1d(axs)[1:]:
        ax.set_ylabel('')
    handles, labels = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0.12, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, 'figA_chain_all_c')
    notes['E1'] = stats
    return panel


def e2(root, out, notes):
    rows = load(root, 'tgap')
    if not rows:
        return None, None
    K = lambda r: int(r['rec']['surveys'])
    s_ = lambda r: int(r['rec']['budget'])
    g = by(rows, K, s_, threshold, arm)
    keys = sorted({k[:3] for k in g})
    V2F = v2_found(root)
    lines, tex = [], []
    for (Kk, ss, c) in keys:
        v2, vb = tg.v2(c, Kk, ss), tg.v2_bar(c, Kk, ss)
        cell = {}
        for name in ('joint_Z1', 'joint_Z2', 'joint_Z4', 'lagrangian', 'eg'):
            rs = g.get((Kk, ss, c, name), [])
            vals = [final_r(r) for r in rs]
            feas = sum(np.isfinite(vals))
            reach = sum(1 for v in vals if v >= vb - 1e-6)
            cell[name] = dict(n=len(rs), feas=feas, med=q(vals, 50), reach=reach)
        lag = g.get((Kk, ss, c, 'lagrangian'), [])
        eps = [max(e for e in (r['rec'].get('eps_lo'), r['rec'].get('eps_hi')) if e is not None)
               for r in lag if r['rec'].get('eps_lo') is not None]
        cert = [r['rec'].get('certificate') for r in lag if r['rec'].get('certificate') is not None]
        pol = [r['rec'].get('polish_J_r') for r in lag if r['rec'].get('polish_J_r') is not None]
        egr = g.get((Kk, ss, c, 'eg'), [])
        found = [v for v in (final_r(r) for r in g.get((Kk, ss, c, 'joint_Z2'), [])) if np.isfinite(v)]
        found += [V2F[(Kk, ss, c)]] if (Kk, ss, c) in V2F else []
        lines.append(dict(K=Kk, s=ss, c=c, v2=v2, v2_found=max(found) if found else np.nan, v2_bar=vb, cell=cell,
                          eps_max=max(eps) if eps else np.nan,
                          cert_med=q(cert, 50), polish_med=q(pol, 50),
                          eg_avg_med=q([r['rec'].get('avg_J_r') for r in egr], 50),
                          eg_viol_max=max([r['rec'].get('avg_violation', 0) for r in egr] or [np.nan])))

    def fmt(x, nd=4):
        if x is None or not np.isfinite(x):
            return '—'
        return f'{x:.2e}' if 0 < abs(x) < 10 ** -nd else f'{x:.{nd}f}'

    def jg(cl):
        if cl['n'] == 0:
            return '—'
        if cl['feas'] < cl['n']:
            return f"infeasible {cl['n'] - cl['feas']}/{cl['n']}" + (f"; {fmt(cl['med'])}" if cl['feas'] else '')
        return fmt(cl['med'])

    def v2cell(L, nd=4, tex=False):
        if np.isfinite(L['v2']):
            return fmt(L['v2'], nd)
        return (r'$\ge$' if tex else '≥ ') + fmt(L['v2_found'], nd) + (r'$^\dagger$' if tex else ' †')

    md = ['| K | s | c | v2 (exact) | v2_bar | joint greedy \\|Z\\|=1 | \\|Z\\|=2 | \\|Z\\|=4 [reach v2_bar] | Lagrangian + bit | '
          '+ polish | max eps at endpoints | certificate (median) | EG average [max violation] |',
          '|' + '---|' * 14]
    for L in lines:
        cl = L['cell']
        md.append(f"| {L['K']} | {L['s']} | {L['c']:g} | {v2cell(L)} | {fmt(L['v2_bar'])} | {jg(cl['joint_Z1'])} | "
                  f"{jg(cl['joint_Z2'])} | {jg(cl['joint_Z4'])} [{cl['joint_Z4']['reach']}/{cl['joint_Z4']['n']}] | "
                  f"{fmt(cl['lagrangian']['med'])} | {fmt(L['polish_med'])} | {fmt(L['eps_max'], 2)} | {fmt(L['cert_med'], 2)} | "
                  f"{fmt(L['eg_avg_med'])} [{fmt(L['eg_viol_max'], 2)}] |")
    md.append('\n† No closed form for 0 < s < K - 1: the budget lets in-budget runs of one cue share a scanning '
              'stretch with scan-everything runs of the other, kept apart by the hidden scan counter rather than '
              'the memory. The entry is the best policy in Pi found (joint greedy with |Z| = 2: the 50 runs here and '
              '600 more starts, tgap_search.py), a lower bound on v2.')
    (out / 'table1_tgap.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    main = [L for L in lines if L['c'] == 0.75]
    tex = [r'\begin{tabular}{rrcccccc}', r'\toprule',
           r'$K$ & $s$ & $v_2$ & $\bar v_2$ & JG $|\mathcal Z|{=}2$ & JG $|\mathcal Z|{=}4$ [reach $\bar v_2$] & Lagrangian + bit & $\max\epsilon$ \\',
           r'\midrule']
    for L in main:
        cl = L['cell']
        tex.append(f"{L['K']} & {L['s']} & {v2cell(L, tex=True)} & {fmt(L['v2_bar'], 2)} & {jg(cl['joint_Z2'])} & "
                   f"{jg(cl['joint_Z4'])} [{cl['joint_Z4']['reach']}/{cl['joint_Z4']['n']}] & {fmt(cl['lagrangian']['med'])} & "
                   f"{fmt(L['eps_max'], 2)} \\\\")
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'table1_tgap.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    notes['E2'] = lines

    def polygon(ax):
        cloud = load(root, 'cloud')
        pts = np.array([p for r in cloud for p in r['log'].get('points', [])]) if cloud else np.zeros((0, 2))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=2, color=MUTED, alpha=0.25, lw=0, label='random policies in $\\Pi$', rasterized=True)
        cgrid = np.linspace(0, 1, 201)
        ax.plot(cgrid, [tg.v2(cc, 2, 0) for cc in cgrid], color=INK2, ls='--', lw=1.0, label='$v_2(c)$: best in $\\Pi$')
        ax.plot([0, 1], [1, 0], color=INK2, ls=':', lw=1.0, label=r'$\bar v_2(c) = 1-c$')
        c = 0.75
        jg_rows = sorted(g.get((2, 0, c, 'joint_Z2'), []), key=lambda r: r['job']['seed'])
        if jg_rows:
            lg = jg_rows[0]['log']
            ax.plot([l['J_phi'] for l in lg], [l['J_r'] for l in lg], color=BLUE, lw=1.0, marker='o', ms=2.5,
                    label='joint greedy iterates (one seed)')
        lag_rows = sorted(g.get((2, 0, c, 'lagrangian'), []), key=lambda r: r['job']['seed'])
        if lag_rows:
            r = lag_rows[0]['rec']
            lo, hi = r['J_lo'], r['J_hi']
            ax.plot([lo['phi'], hi['phi']], [lo['r'], hi['r']], color=AQUA, lw=1.5, label='bracket endpoints and mixture')
            ax.scatter([lo['phi'], hi['phi']], [lo['r'], hi['r']], s=18, color=AQUA, edgecolor='white', lw=0.8, zorder=5)
            ax.scatter([r['J_phi']], [r['J_r']], s=60, marker='*', color=AQUA, edgecolor=INK, lw=0.5, zorder=6)
        ax.axvline(c, color=INK2, lw=0.8)
        ax.text(c + 0.015, 0.45, f'$c={c:g}$', color=INK2, fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(r'$J_\varphi$')
        ax.set_ylabel(r'$J_r$')
        ax.set_title('Survey maze, $K=2$', loc='left')

    def convergence(ax, Kk=4, ss=0, c=0.75):
        xs = np.unique(np.round(np.logspace(0, 3, 160))).astype(float)
        drawn = False
        for name, col, lab, ls in (('joint_Z2', BLUE, 'joint greedy, $|Z|=2$', '-'),
                                   ('joint_Z4', ORANGE, 'joint greedy, $|Z|=4$', '-'),
                                   ('lagrangian', AQUA, 'Lagrangian, $|Z|=2\\times2$', '-')):
            rs = g.get((Kk, ss, c, name), [])
            if not rs:
                continue
            med, lo, hi, share = curve_band(rs, xs)
            ok = share >= 0.5
            drawn |= bool(ok.any())
            ax.plot(xs[ok], med[ok], color=col, ls=ls, label=lab)
            ax.fill_between(xs[ok], lo[ok], hi[ok], color=col, alpha=0.12, lw=0)
        ax.axhline(tg.v2_bar(c, Kk, ss), color=INK2, ls=':', lw=1.0)
        ax.axhline(tg.v2(c, Kk, ss), color=INK2, ls='--', lw=1.0)
        ax.text(1.02, tg.v2_bar(c, Kk, ss), r'$\bar v_2$', color=INK2, fontsize=7, va='center',
                transform=ax.get_yaxis_transform())
        ax.text(1.02, tg.v2(c, Kk, ss), r'$v_2$', color=INK2, fontsize=7, va='center', transform=ax.get_yaxis_transform())
        if drawn:
            ax.set_xscale('log')
        else:
            ax.text(0.5, 0.5, 'no sweep logs (log.json) found', transform=ax.transAxes, ha='center', fontsize=7, color=INK2)
        ax.set_xlabel('sweeps')
        ax.set_ylabel(r'$J_r$ at $J_\varphi \geq c$')
        ax.set_title(f'Survey maze, $K={Kk}$', loc='left')
    return polygon, convergence


def fig3(root, out, notes):
    panel_e1 = e1(root, out, notes)
    polygon, convergence = e2(root, out, notes)
    present = [p for p in (panel_e1, polygon, convergence) if p is not None]
    if not present:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.25))
    if panel_e1:
        panel_e1(axs[0], 0.75)
    if polygon:
        polygon(axs[1])
        convergence(axs[2])
    for ax, lab in zip(axs, 'abc'):
        ax.set_title(f'({lab}) ' + ax.get_title(loc='left'), loc='left')
        if ax.get_legend():
            ax.get_legend().remove()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, marker='o', ms=4, mec='white', label='joint greedy, memory budget only'),
               Line2D([], [], color=ORANGE, marker='s', ms=4, mec='white', label='joint greedy, one spare bit'),
               Line2D([], [], color=AQUA, marker='D', ms=4, mec='white', label='Lagrangian method, one extra bit'),
               Line2D([], [], color=INK2, ls='--', lw=1.0, label=r'$v_2$ (exact, best in $\Pi$)'),
               Line2D([], [], color=INK2, ls=':', lw=1.0, label=r'$\bar v_2$ (exact, mixtures)'),
               Line2D([], [], color=MUTED, marker='.', ls='none', ms=5, label=r'random policies in $\Pi$ (b)'),
               Line2D([], [], color=AQUA, marker='*', ls='none', ms=8, mec=INK, mew=0.5, label=r'$\pi_\theta$ (b)')]
    fig.tight_layout(w_pad=1.6, rect=(0, 0.12, 1, 1))
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    save(fig, out, 'fig3_gap')


def fig4(main_root, out, notes):
    rows = load(main_root, 'tmaze_main')
    rows = [r for r in rows if r['rec'].get('variant') == 'logic' and r['rec'].get('objective') == 'P1']
    if not rows:
        return
    g = by(rows, lambda r: r['rec']['Z'], lambda r: r['rec']['margin'], lambda r: r['rec']['alpha'], lambda r: r['rec']['slack'])
    slacks = sorted({k[3] for k in g})
    solved = {}
    for key in ((2, 1, 0.9), (2, 0, 0.9), (2, 1, 1.0), (2, 0, 1.0), (1, 1, 0.9), (1, 1, 1.0), (1, 0, 0.9), (1, 0, 1.0)):
        solved[key] = [float(np.mean([r['rec']['J_phi'] >= 1 - 1e-9 for r in g[key + (sl,)]])) if g.get(key + (sl,)) else np.nan
                       for sl in slacks]
    notes['E3'] = dict(slacks=slacks, solved={str(k): v for k, v in solved.items()})
    st = load(main_root, 'start_ablation')
    if st:
        gs = by(st, lambda r: r['rec']['Z'], lambda r: r['rec']['start'], lambda r: r['rec']['ties'])
        md = ['| \\|Z\\| | start | ties | runs solved | memory bits (mean) |', '|---|---|---|---|---|']
        for (Z, s_, t), rs in sorted(gs.items()):
            bits = [r['rec'].get('memory_bits') for r in rs if r['rec'].get('memory_bits') is not None]
            md.append(f"| {Z} | {s_} | {t} | {sum(r['rec']['J_phi'] >= 1 - 1e-9 for r in rs)}/{len(rs)} | "
                      f"{np.mean(bits) if bits else float('nan'):.2f} |")
        (out / 'tableA_start.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def table3_tmaze(main_root, out, notes):
    rows = [r for r in load(main_root, 'tmaze_main') if r['rec'].get('variant') == 'logic' and r['rec'].get('objective') == 'P1']
    if not rows:
        return
    g = by(rows, lambda r: (r['rec']['Z'], r['rec']['alpha'], r['rec']['margin']), lambda r: r['rec']['slack'])
    keys = sorted({k[0] for k in g}, key=lambda k: (-k[0], k[1], k[2]))
    slacks = sorted({k[1] for k in g})
    Ls = sorted({r['rec']['L'] for r in rows})
    md = ['| \\|Z\\| | alpha | margin | ' + ' | '.join(f'slack {s}' for s in slacks) + ' | sweeps (median) |',
          '|' + '---|' * (len(slacks) + 4)]
    tex = [r'\begin{tabular}{ccc' + 'c' * len(slacks) + 'c}', r'\toprule',
           r'& & & \multicolumn{' + str(len(slacks)) + r'}{c}{slack} & \\',
           r'\cmidrule(lr){4-' + str(3 + len(slacks)) + '}',
           r'$|\mathcal Z|$ & $\alpha$ & margin & ' + ' & '.join(str(s) for s in slacks) + r' & sweeps \\', r'\midrule']
    for k in keys:
        Z, a, m = k
        shares, sw = [], []
        for s_ in slacks:
            rs = g.get((k, s_), [])
            shares.append(np.mean([r['rec']['J_phi'] >= 1 - 1e-9 for r in rs]) if rs else np.nan)
            sw += [r['rec'].get('sweeps') for r in rs]
        cells = ['—' if not np.isfinite(x) else f'{x:.2f}' for x in shares]
        a_s = '1 (lazy)' if a == 1.0 else f'{a:g}'
        md.append(f'| {Z} | {a_s} | {m} | ' + ' | '.join(cells) + f' | {q(sw, 50):.0f} |')
        tex.append(f'{Z} & {a_s} & {m} & ' + ' & '.join(cells) + f' & {q(sw, 50):.0f}' + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    n_per = len(rows) // max(1, len(keys) * len(slacks))
    md += ['', f'Share of runs with J_phi = 1, pooled over corridor lengths {Ls} ({n_per} runs per cell).']
    (out / 'table3_tmaze.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'table3_tmaze.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')


TMAZE_CURVE_SLACK, TMAZE_CURVE_MARGIN = 2, 0


def fig4_tmaze(main_root, out, notes):
    rows = [r for r in load(main_root, 'tmaze_main') if r['rec'].get('variant') == 'logic' and r['rec'].get('objective') == 'P1'
            and r['rec']['margin'] == TMAZE_CURVE_MARGIN and r['rec']['slack'] == TMAZE_CURVE_SLACK and r['log']]
    if not rows:
        return
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    Ls = sorted({r['rec']['L'] for r in rows})
    fig, axs = plt.subplots(1, len(Ls), figsize=(1.75 * len(Ls), 2.25), sharey=True, sharex=True, squeeze=False)
    axs = axs[0]
    series = (((2, 0.9), BLUE, '-'), ((2, 1.0), BLUE, '--'), ((1, 0.9), ORANGE, '-'), ((1, 1.0), ORANGE, '--'))
    stats = {}
    for ax, L in zip(axs, Ls):
        pick = [r for r in rows if r['rec']['L'] == L]
        n = max(len(r['log']) for r in pick)
        for (Z, a), col, ls in series:
            rs = [r for r in pick if r['rec']['Z'] == Z and r['rec']['alpha'] == a]
            if not rs:
                continue
            M = np.empty((len(rs), n))
            for k, r in enumerate(rs):
                v = [l['J_phi'] for l in r['log']]
                M[k, :len(v)] = v
                M[k, len(v):] = v[-1]
            med = np.median(M, 0)
            ax.plot(np.arange(n), med, color=col, ls=ls, lw=1.4)
            ax.fill_between(np.arange(n), np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=col, alpha=0.12, lw=0)
            stop = int(np.median([len(r['log']) - 1 for r in rs]))
            ax.plot([stop], [med[stop]], marker='o', ms=3.2, color=col, mec='white', mew=0.6)
            best = 1.0 if Z == 2 else 0.5
            stats[(L, Z, a)] = (sum(r['rec']['J_phi'] >= best - 1e-6 for r in rs), len(rs), q([r['rec']['sweeps'] for r in rs], 50))
        for y in (1.0, 0.5):
            ax.axhline(y, color=INK2, ls=':', lw=0.8, zorder=0)
        ax.set_title(f'$L={L}$', loc='left')
        ax.set_xlabel('sweeps')
        ax.set_ylim(-0.03, 1.06)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    axs[0].set_ylabel('$J_\\varphi$')
    axs[-1].text(1.03, 1.0, r'best, $|Z|=2$', transform=axs[-1].get_yaxis_transform(), fontsize=6.5, va='center', color=INK2)
    axs[-1].text(1.03, 0.5, r'best, $|Z|=1$', transform=axs[-1].get_yaxis_transform(), fontsize=6.5, va='center', color=INK2)
    handles = [Line2D([], [], color=BLUE, ls='-', label=r'$|Z|=2$, $\alpha=0.9$'),
               Line2D([], [], color=BLUE, ls='--', label=r'$|Z|=2$, $\alpha=1$ (lazy)'),
               Line2D([], [], color=ORANGE, ls='-', label=r'$|Z|=1$, $\alpha=0.9$'),
               Line2D([], [], color=ORANGE, ls='--', label=r'$|Z|=1$, $\alpha=1$ (lazy)'),
               Line2D([], [], color=INK2, marker='o', ls='none', ms=3.2, mec='white', label='median stopping sweep')]
    fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 0.93, 1))
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=6.3, bbox_to_anchor=(0.47, 0.0), handlelength=1.8,
               columnspacing=1.1)
    save(fig, out, 'fig4_tmaze')
    notes['tmaze_curves'] = {f'L={L}|Z={Z}|alpha={a:g}': v for (L, Z, a), v in stats.items()}


def fig5_rock(root, out, notes):
    rock = load(root, 'rock')
    if not rock:
        return
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.75))
    cfgs = {(int(r['rec'].get('rocks', 0)), threshold(r)) for r in rock}
    cfg = (5, 0.9) if (5, 0.9) in cfgs else (3, 0.8)
    rs_all = [r for r in rock if int(r['rec'].get('rocks', 0)) == cfg[0] and abs(threshold(r) - cfg[1]) < 1e-9]
    xs = np.unique(np.round(np.logspace(0, 3.3, 220))).astype(float)
    arms = {a: [r for r in rs_all if arm(r) == a] for a in ('joint_Z2', 'joint_Z4', 'lagrangian', 'eg')}
    ends = []
    for name, col, ls, lab in (('joint_Z2', BLUE, '-', 'joint greedy, $|Z|=2$'), ('joint_Z4', BLUE, '--', 'joint greedy, $|Z|=4$'),
                               ('lagrangian', AQUA, '-', 'Lagrangian method'), ('eg', MUTED, '-', 'exp. gradient (best pair)')):
        rs = arms[name]
        if not rs:
            continue
        med, lo, hi, share = curve_band(rs, xs)
        ok = share >= 0.5
        ax.plot(xs[ok], med[ok], color=col, ls=ls, label=lab)
        ax.fill_between(xs[ok], lo[ok], hi[ok], color=col, alpha=0.12, lw=0)
        ends.append(med[ok][-1] if ok.any() else np.nan)
    jg = arms['joint_Z2']
    if jg:
        finals = [final_r(r) for r in jg]
        m = q([r['rec'].get('sweeps') for r in jg], 50)
        ks = np.arange(1, len(finals) + 1)
        bx = ks * m
        by_ = [expected_best_of_k(finals, k, draws=2000) for k in ks]
        keep = bx <= xs[-1]
        ax.plot(bx[keep], np.array(by_)[keep], color=INK2, ls=':', lw=1.2, label='best of $k$ joint-greedy runs')
        ends.append(by_[min(len(by_), int(keep.sum())) - 1])
    top = np.nanmax(ends) if ends else 1.0
    base = q([final_r(r) for r in jg], 50) if jg else top
    ax.set_ylim(base - 2.0 * (top - base) - 0.2, top + 0.6 * (top - base) + 0.05)
    ax.set_xscale('log')
    ax.set_xlabel('sweeps')
    ax.set_ylabel('$J_r$ of the returned policy')
    ax.set_title(f'RockSample, {cfg[0]} rocks, $c={cfg[1]:g}$', loc='left')
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    save(fig, out, 'fig5_rock')
    notes['fig5_rock_cfg'] = cfg


def fig_tmaze_slack(root, out, notes):
    rows = load(root, 'tmaze_slack')
    if not rows:
        return
    from matplotlib.ticker import MaxNLocator
    cpi_rows = [r for r in rows if r['job']['kind'] == 'cpi']
    route = [r for r in rows if r['job']['kind'] == 'route']
    g = by(cpi_rows, lambda r: r['rec']['L'], lambda r: r['rec']['alpha'], lambda r: r['rec']['slack'])
    Ls = sorted({k[0] for k in g})
    Ws = sorted({k[2] for k in g})
    share = {k: float(np.mean([r['rec']['J_phi'] >= 1 - 1e-9 for r in rs])) for k, rs in g.items()}
    gr = by(route, lambda r: r['rec']['L'], lambda r: r['rec']['slack'])
    exists = {k: (sum(r['rec'].get('route_exists', False) for r in rs), len(rs)) for k, rs in gr.items()}
    w_exist = [W for W in Ws if all(exists.get((L, W), (0, 1))[0] == exists.get((L, W), (0, 1))[1] > 0 for L in Ls)]
    w0 = min(w_exist) if w_exist else None

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))
    shades = [1.0, 0.75, 0.5, 0.3][:len(Ls)][::-1]
    for L, a in zip(Ls, shades):
        ys = [share.get((L, 0.9, W), np.nan) for W in Ws]
        ax.plot(Ws, ys, color=BLUE, alpha=max(a, 0.3), marker='o', ms=3, mec='white', mew=0.5, lw=1.3, label=f'$L={L}$')
    pooled = [np.mean([share.get((L, 0.9, W), np.nan) for L in Ls]) for W in Ws]
    ax.plot(Ws, pooled, color=INK2, lw=1.0, ls='--', label='all $L$')
    lazy = [np.mean([share.get((L, 1.0, W), np.nan) for L in Ls]) for W in Ws]
    ax.plot(Ws, lazy, color=ORANGE, lw=1.0, ls=':', label=r'all $L$, $\alpha=1$ (lazy)')
    if w0 is not None:
        ax.axvline(w0, color=INK2, lw=0.8, ls=':')
        ax.text(w0 - 0.2, 0.64, 'memoryless optimum exists', fontsize=6, color=INK2, rotation=90,
                ha='right', va='center')
    ax.set_xlabel('slack $W$')
    ax.set_ylabel(r'share of runs with $J_\varphi=1$')
    ax.set_ylim(-0.03, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(r'(P1), $|Z|=1$, start margin 1', loc='left')
    ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    save(fig, out, 'figA_tmaze_slack')

    md = ['| | ' + ' | '.join(f'W={W}' for W in Ws) + ' |', '|' + '---|' * (len(Ws) + 1)]
    for L in Ls:
        md.append(f'| L={L}, alpha=0.9 | ' + ' | '.join(f"{share.get((L, 0.9, W), np.nan):.2f}" for W in Ws) + ' |')
    md.append('| all L, alpha=0.9 | ' + ' | '.join(f'{v:.2f}' for v in pooled) + ' |')
    md.append('| all L, alpha=1 (lazy) | ' + ' | '.join(f'{v:.2f}' for v in lazy) + ' |')
    md.append('| route policy reaches J_phi = 1 | ' + ' | '.join(
        f"{sum(exists.get((L, W), (0, 0))[0] for L in Ls)}/{sum(exists.get((L, W), (0, 0))[1] for L in Ls)}" for W in Ws) + ' |')
    md += ['', 'Share of (P1) runs with |Z| = 1 and start margin 1 that reach J_phi = 1 (50 seeds per cell); last row: '
           'layouts on which the memoryless route policy (tmaze_route.py) reaches J_phi = 1.']
    (out / 'tableA_tmaze_slack.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    notes['tmaze_slack'] = dict(W=Ws, pooled=pooled, lazy=lazy, w_exists=w0,
                                by_L={str(L): [share.get((L, 0.9, W), np.nan) for W in Ws] for L in Ls})


ROCK_ARMS = (('joint_Z2', 'Joint greedy, |Z|=2'), ('joint_Z4', 'Joint greedy, |Z|=4'),
             ('restarts', 'Joint greedy, best of k (same sweeps)'), ('naive_Z2', 'Reward-greedy step + line search'),
             ('lagrangian', 'Lagrangian + bit (best pair)'), ('polish', 'Lagrangian + polish'),
             ('eg_avg', 'Exp. gradient, average of iterates'), ('eg_pair', 'Exp. gradient, best pair'))


def expected_best_of_k(vals, k, draws=20000, seed=0):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return np.nan
    k = int(min(max(1, k), len(vals)))
    rng = np.random.default_rng(seed)
    return float(np.mean([rng.choice(vals, size=k, replace=False).max() for _ in range(draws)]))


def e4(root, out, notes):
    rows = load(root, 'rock')
    if not rows:
        return
    rocks = lambda r: int(r['rec']['rocks'])
    g = by(rows, rocks, threshold, arm)
    configs = sorted({k[:2] for k in g})
    seed = lambda r: r['job']['seed']
    res, curves = {}, {}
    for cfg in configs:
        base = {seed(r): final_r(r) for r in g.get(cfg + ('joint_Z2',), [])}
        sweeps_jg = [r['rec'].get('sweeps') for r in g.get(cfg + ('joint_Z2',), [])]
        lag = g.get(cfg + ('lagrangian',), [])
        sweeps_lag = [r['rec'].get('sweeps') for r in lag]
        k = int(round(q(sweeps_lag, 50) / q(sweeps_jg, 50))) if lag and sweeps_jg else 1
        vals = {}
        vals['joint_Z2'] = base
        vals['joint_Z4'] = {seed(r): final_r(r) for r in g.get(cfg + ('joint_Z4',), [])}
        vals['naive_Z2'] = {seed(r): final_r(r) for r in g.get(cfg + ('naive_Z2',), [])}
        vals['lagrangian'] = {seed(r): final_r(r) for r in lag}
        vals['polish'] = {seed(r): r['rec'].get('polish_J_r', np.nan) for r in lag}
        eg = g.get(cfg + ('eg',), [])
        vals['eg_avg'] = {seed(r): r['rec'].get('avg_J_r', np.nan) for r in eg}
        vals['eg_pair'] = {seed(r): final_r(r) for r in eg}
        cell = {}
        b_mean, b_ci, b_n = mean_ci(list(base.values()))
        cell['joint_Z2'] = (b_mean, b_ci, b_n, 'abs')
        for name in ('joint_Z4', 'naive_Z2', 'lagrangian', 'polish', 'eg_avg', 'eg_pair'):
            d = [vals[name][s] - base[s] for s in vals[name] if s in base]
            cell[name] = mean_ci(d) + ('diff',)
        cell['restarts'] = (expected_best_of_k(list(base.values()), k) - b_mean, np.nan, k, 'restarts')
        theta = [r['rec'].get('pair_theta') for r in lag if r['rec'].get('pair_theta') is not None]
        degenerate = [s for s, r in ((seed(r), r) for r in lag)
                      if r['rec'].get('pair_theta') is not None and (r['rec']['pair_theta'] < 1e-6 or r['rec']['pair_theta'] > 1 - 1e-6)]
        gain_deg = [vals['lagrangian'][s] - base[s] for s in degenerate if s in base]
        gain_mix = [vals['lagrangian'][s] - base[s] for s in vals['lagrangian'] if s in base and s not in degenerate]
        viol = [r['rec'].get('avg_violation', 0) for r in eg]
        res[cfg] = dict(cell=cell, k=k, sweeps=dict(joint=q(sweeps_jg, 50), lagrangian=q(sweeps_lag, 50),
                                                       eg=q([r['rec'].get('sweeps') for r in eg], 50),
                                                       joint_Z4=q([r['rec'].get('sweeps') for r in g.get(cfg + ('joint_Z4',), [])], 50)),
                        degenerate=len(degenerate), n_lag=len(lag), gain_deg=q(gain_deg, 50), gain_mix=q(gain_mix, 50),
                        eg_viol_max=max(viol) if viol else np.nan,
                        lag_phi_err=max([abs(r['rec']['J_phi'] - threshold(r)) for r in lag] or [np.nan]),
                        best_pair_vs_best_single=(max([v for v in vals['lagrangian'].values() if np.isfinite(v)] or [np.nan])
                                                  - max([v for v in base.values() if np.isfinite(v)] or [np.nan])))
        curves[cfg] = {name: g.get(cfg + (name,), []) for name in ('joint_Z2', 'joint_Z4', 'lagrangian', 'eg')}
    head = '| Method | ' + ' | '.join(f'{k} rocks, c={c:g}' for k, c in configs) + ' |'
    md = [head, '|---|' + '---|' * len(configs)]
    for name, lab in ROCK_ARMS:
        cells = []
        for cfg in configs:
            m, h, n, kind = res[cfg]['cell'][name]
            if not np.isfinite(m):
                cells.append('—')
            elif kind == 'abs':
                cells.append(f'{m:.2f} ± {h:.2f}')
            elif kind == 'restarts':
                cells.append(f'{m:+.2f} (k={n})')
            else:
                cells.append(f'{m:+.2f} ± {h:.2f}' + ('' if n >= 50 else f' (n={n})'))
        md.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
    md.append('')
    md.append('First row: mean J_r ± 95% CI. Other rows: paired difference from the first row (same seeds). '
              '"best of k": expected best of k joint-greedy runs, k = median Lagrangian sweeps / median joint-greedy sweeps.')
    md.append('')
    md.append('| config | sweeps: joint | joint Z4 | Lagrangian | EG | degenerate pairs (theta in {0,1}) | gain, degenerate | gain, mixed | '
              'EG max violation | max abs(J_phi - c), Lagrangian | best pair minus best single run |')
    md.append('|---|' + '---|' * 10)
    for cfg in configs:
        R = res[cfg]
        sw = R['sweeps']
        md.append(f"| {cfg[0]} rocks, c={cfg[1]:g} | {sw['joint']:.0f} | {sw['joint_Z4']:.0f} | {sw['lagrangian']:.0f} | {sw['eg']:.0f} | "
                  f"{R['degenerate']}/{R['n_lag']} | {R['gain_deg']:+.3f} | {R['gain_mix']:+.3f} | {R['eg_viol_max']:.3f} | "
                  f"{R['lag_phi_err']:.1e} | {R['best_pair_vs_best_single']:+.3f} |")
    (out / 'table2_rock.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    tex = [r'\begin{tabular}{l' + 'c' * len(configs) + '}', r'\toprule',
           'Method & ' + ' & '.join(f'{k}r, $c={c:g}$' for k, c in configs) + r' \\', r'\midrule']
    for line in md[2:2 + len(ROCK_ARMS)]:
        parts = [p.strip() for p in line.strip('|').split('|')]
        tex.append(' & '.join(p.replace('±', r'$\pm$').replace('|Z|', r'$|\mathcal Z|$') for p in parts) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    (out / 'table2_rock.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    notes['E4'] = {f'{k}r c={c:g}': dict(k=res[(k, c)]['k'], sweeps=res[(k, c)]['sweeps'], degenerate=res[(k, c)]['degenerate'],
                                         n_lag=res[(k, c)]['n_lag'], gain_deg=res[(k, c)]['gain_deg'],
                                         gain_mix=res[(k, c)]['gain_mix'], eg_viol_max=res[(k, c)]['eg_viol_max'],
                                         best_pair_vs_best_single=res[(k, c)]['best_pair_vs_best_single'],
                                         cells={n: res[(k, c)]['cell'][n][:3] for n, _ in ROCK_ARMS})
                   for (k, c) in configs}
    ks = sorted({k for k, _ in configs})
    ncol = max(sum(1 for k2, _ in configs if k2 == k) for k in ks)
    fig, axs = plt.subplots(len(ks), ncol, figsize=(2.3 * ncol, 1.9 * len(ks) + 0.4), sharex=True, sharey=True, squeeze=False)
    place = {}
    for i, k in enumerate(ks):
        for j, cfg in enumerate([cf for cf in configs if cf[0] == k]):
            place[cfg] = axs[i, j]
    for ax in axs.flat:
        if ax not in place.values():
            ax.set_visible(False)
    xs = np.unique(np.round(np.logspace(0, 3.2, 200))).astype(float)
    for cfg, ax in place.items():
        base = {r['job']['seed']: final_r(r) for r in curves[cfg]['joint_Z2']}
        for name, col, lab in (('joint_Z2', BLUE, 'joint greedy, $|Z|=2$'), ('joint_Z4', ORANGE, 'joint greedy, $|Z|=4$'),
                               ('lagrangian', AQUA, 'Lagrangian + bit'), ('eg', MUTED, 'exp. gradient, best pair')):
            rs = [r for r in curves[cfg][name] if r['job']['seed'] in base]
            if not rs:
                continue
            M = []
            for r in rs:
                cv = incumbent_curve(r)
                s = np.array([a for a, _ in cv], float)
                v = np.array([b for _, b in cv], float) - base[r['job']['seed']]
                idx = np.searchsorted(s, xs, side='right') - 1
                row = np.full(len(xs), np.nan)
                row[idx >= 0] = v[idx[idx >= 0]]
                M.append(row)
            M = np.array(M)
            share = np.mean(np.isfinite(M), 0)
            ok = share >= 0.5
            with np.errstate(all='ignore'):
                ax.plot(xs[ok], np.nanmedian(M, 0)[ok], color=col, label=lab)
                ax.fill_between(xs[ok], np.nanpercentile(M, 25, 0)[ok], np.nanpercentile(M, 75, 0)[ok], color=col, alpha=0.12, lw=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale('log')
        ax.set_title(f'{cfg[0]} rocks, $c={cfg[1]:g}$')
        ax.set_xlim(1, 10 ** 3.2)
        ax.set_ylim(-1.0, 0.6)
    for ax in axs[-1]:
        ax.set_xlabel('sweeps')
    for ax in axs[:, 0]:
        ax.set_ylabel('$J_r$ minus final\njoint greedy ($|Z|=2$)')
    handles, labels = next(iter(place.values())).get_legend_handles_labels()
    fig.tight_layout(w_pad=0.8, rect=(0, 0.1 / len(ks) + 0.02, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, 'figA_rock_curves')


def e5(root, out, notes):
    rows = load(root, 'threshold')
    if not rows:
        return
    g = by(rows, lambda r: r['job']['task'], threshold)
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.0))
    for task, col in (('rocksample3_tight', BLUE), ('tgap_K3_s0', ORANGE)):
        ks = sorted(c for (t, c) in g if t == task)
        if not ks:
            continue
        kap = [q([r['rec'].get('kappa') for r in g[(task, c)]], 50) for c in ks]
        wid = [q([r['rec'].get('width') for r in g[(task, c)]], 50) for c in ks]
        cer = [q([r['rec'].get('certificate') for r in g[(task, c)]], 50) for c in ks]
        lab = 'RockSample, 3 rocks' if task.startswith('rock') else 'survey maze, $K=3$'
        axs[0].plot(kap, wid, color=col, marker='o', ms=3.5, mec='white', label=lab)
        axs[1].plot(kap, cer, color=col, marker='o', ms=3.5, mec='white', label=lab)
    for ax, yl in zip(axs, (r"bracket width $\lambda''-\lambda'$", 'certificate term')):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\tilde\kappa = J_\varphi(\tilde\pi) - c$')
        ax.set_ylabel(yl)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, 'figA_threshold')


def e_memstart(root, out, notes):
    rows = load(root, 'memstart')
    if not rows:
        return
    starts = (('copy', 'random', 'copy'), ('copycoin', 'random', 'copy + coin at $t{=}1$'),
              ('dirichlet', 'index', 'Dirichlet'), ('uniform', 'index', 'uniform'))
    def group(task):
        return 'T-maze' if task.startswith('tmaze') else f"survey maze, $K={task.split('_')[1][1:]}$"
    g = by(rows, lambda r: group(r['job']['task']), lambda r: (r['job']['start'], r['job']['ties']))
    groups = sorted({k[0] for k in g}, key=lambda x: (x != 'T-maze', len(x), x))
    md = ['| maze | ' + ' | '.join(lab.replace('$t{=}1$', 't = 1') for _, _, lab in starts) + ' |', '|' + '---|' * (len(starts) + 1)]
    tex = [r'\begin{tabular}{l' + 'c' * len(starts) + '}', r'\toprule',
           'Maze & ' + ' & '.join(lab for _, _, lab in starts) + r' \\', r'\midrule']
    cells = {}
    for grp in groups:
        line = []
        for st, ti, _ in starts:
            rs = g.get((grp, (st, ti)), [])
            solved = sum(1 for r in rs if r['rec']['J_phi'] >= 1 - 1e-6)
            cells[(grp, st)] = (solved, len(rs), q([r['rec'].get('memory_bits') for r in rs], 50))
            line.append(f'{solved}/{len(rs)}' if rs else '—')
        name = grp + (' (8 layouts)' if grp == 'T-maze' else '')
        md.append(f"| {name.replace('$', '')} | " + ' | '.join(line) + ' |')
        tex.append(f'{name} & ' + ' & '.join(line) + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}']
    md += ['', '(P1) with |Z| = 2: runs that reach J_phi = 1, i.e. store the cue and use it. T-maze: the cue-key-door maze, '
           'corridor 10 or 20, slack 0 or 1, margin 0 or 1, 50 seeds each.']
    (out / 'tableA_memstart.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'tableA_memstart.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    notes['memstart'] = {f'{grp}|{st}': v for (grp, st), v in cells.items()}


def e_eglong(root, out, notes):
    rows = load(root, 'eg_long') + [r for r in load(root, 'rock') if r['job']['task'] == 'rocksample3_tight'
                                    and r['job']['kind'] in ('eg', 'bisect')]
    if not any(r['job']['exp'] == 'eg_long' for r in rows):
        return
    g = by(rows, threshold, lambda r: (r['job']['kind'], r['job'].get('iters', 16)))
    cs = sorted({k[0] for k in g})
    kinds = [('bisect', 16)] + sorted({k[1] for k in g if k[1][0] == 'eg'}, key=lambda x: x[1])
    md = ['| c | method | oracle calls | runs | returned J_r (median) | J_phi of the average | final multiplier (median) |',
          '|---|---|---|---|---|---|---|']
    res = {}
    for c in cs:
        for kind, K in kinds:
            rs = g.get((c, (kind, K)), [])
            if not rs:
                continue
            if kind == 'bisect':
                val = q([r['rec'].get('pair_value') for r in rs], 50)
                phi = q([r['rec'].get('J_phi') for r in rs], 50)
                lam = q([r['rec'].get('lam_hi') for r in rs], 50)
                name = 'bisection (Theorem 6)'
            else:
                val = q([r['rec'].get('avg_J_r') for r in rs], 50)
                phi = q([r['rec'].get('avg_J_phi') for r in rs], 50)
                lam = q([r['log']['oracle'][-1]['lam'] for r in rs if r['log'].get('oracle')], 50)
                name = 'exp. gradient, average'
            res[(c, kind, K)] = (val, phi, lam, len(rs))
            md.append(f'| {c:g} | {name} | {K} | {len(rs)} | {val:.3f} | {phi:.3f} | {lam:.3g} |')
    md += ['', 'Same inner oracle for both. The bisection returns a mixture meeting c exactly; exponentiated gradient '
           'returns the average of its iterates. Final multiplier: the upper end of the bracket, or the last Hedge multiplier.']
    (out / 'tableA_eg_long.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    notes['eg_long'] = {f'{c:g}|{k}|{K}': v for (c, k, K), v in res.items()}


def checks(root):
    out = dict(runs=0, joint_not_monotone=0, joint_max_drop=0.0, drop_min_lam=np.inf, joint_min_phi_gap=np.inf, max_stage_gap=0.0, max_identity_err=0.0,
               lag_max_phi_err=0.0, lag_curve_check=0.0, lbs_checked=0, lbs_held=0, lbs_gap_max=0.0, lbs_cert_max=0.0)
    for exp in ('chain', 'tgap', 'rock', 'threshold'):
        for r in load(root, exp):
            rec = r['rec']
            if r['job']['kind'] == 'cpi' and rec.get('feasible_start'):
                out['runs'] += 1
                out['joint_not_monotone'] += int(not rec.get('monotone', True))
                out['joint_max_drop'] = max(out['joint_max_drop'], rec.get('max_drop') or 0.0)
                if (rec.get('max_drop') or 0.0) > 1e-10:
                    out['drop_min_lam'] = min(out['drop_min_lam'], rec.get('lam_max') or np.inf)
                if rec.get('min_phi_P2') is not None and np.isfinite(rec['min_phi_P2']):
                    out['joint_min_phi_gap'] = min(out['joint_min_phi_gap'], rec['min_phi_P2'] - threshold(r))
                out['max_stage_gap'] = max(out['max_stage_gap'], rec.get('dual_gap') or 0.0)
                out['max_identity_err'] = max(out['max_identity_err'], rec.get('identity_err') or 0.0)
            if r['job']['kind'] == 'bisect' and rec.get('status') == 'mixed' and rec.get('eps_lo') is not None:
                task = r['job']['task']
                vb = (chain.v2_bar(threshold(r)) if task.startswith('chain') else
                      tg.v2_bar(threshold(r), int(rec['surveys']), int(rec['budget'])) if task.startswith('tgap') else None)
                if vb is not None:
                    th = rec['theta']
                    bound = vb - th * rec['eps_hi'] - (1 - th) * rec['eps_lo'] - rec['certificate']
                    out['lbs_checked'] += 1
                    out['lbs_held'] += int(rec['bracket_J_r'] >= bound - 1e-9)
                    out['lbs_gap_max'] = max(out['lbs_gap_max'], vb - rec['J_r'])
                    out['lbs_cert_max'] = max(out['lbs_cert_max'], rec['certificate'])
            if r['job']['kind'] == 'bisect' and rec.get('J_phi') is not None:
                out['lag_max_phi_err'] = max(out['lag_max_phi_err'], abs(rec['J_phi'] - threshold(r)) if rec.get('status') == 'mixed' else 0.0)
                out['lag_curve_check'] = max(out['lag_curve_check'], rec.get('curve_check') or 0.0)
    return out


def _table(path, cols, rows):
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        md.append('| ' + ' | '.join(f'{r[c]:.4g}' if isinstance(r[c], float) else str(r[c]) for c in cols) + ' |')
    path.write_text('\n'.join(md) + '\n', encoding='utf-8')


def numbers(root, out, notes):
    L = ['# In-text numbers by result (generated by report_paper.py)', '']
    ch = checks(root)

    L += ['## Lemma 2 and Proposition 3: exact stage identity; joint greedy is a feasible ascent', '']
    L.append(f"- Over {ch['runs']} joint-greedy runs: largest stage-identity error {ch['max_identity_err']:.1e}; largest "
             f"stage optimality gap (complementary slackness / (1 + lambda)) {ch['max_stage_gap']:.1e}, for the step that "
             f"randomizes one row; smallest J_phi - c during (P2) {ch['joint_min_phi_gap']:.1e}.")
    L.append(f"- J_r never decreased (relative tolerance 1e-9) after the first feasible iterate in "
             f"{ch['runs'] - ch['joint_not_monotone']}/{ch['runs']} runs; the largest decrease anywhere was "
             f"{ch['joint_max_drop']:.1e}, only in runs whose stage multiplier reached {ch['drop_min_lam']:.1e} or more (round-off "
             f"in the Q-values times the multiplier).")
    if 'E4' in notes:
        L.append('- Without the multiplier (reward-greedy step + line search), RockSample: ' + '; '.join(
            f"{cfg}: {R['cells']['naive_Z2'][0]:+.2f}" for cfg, R in notes['E4'].items()) + ' against joint greedy.')

    L += ['', '## Section 3.3 and Remark 7: person-by-person fixed points and the start', '']
    if 'E3' in notes:
        L.append(f"- Cue-key-door T-maze (P1), share solved by slack {notes['E3']['slacks']}: " +
                 '; '.join(f'{k}: ' + ', '.join(f'{v:.2f}' for v in vs) for k, vs in notes['E3']['solved'].items()))
    if 'tmaze_curves' in notes:
        tc = notes['tmaze_curves']
        def agg(Z, a):
            v = [x for k, x in tc.items() if f'|Z={Z}|' in k and k.endswith(f'alpha={a:g}')]
            return sum(x[0] for x in v), sum(x[1] for x in v), [x[2] for x in v]
        parts = []
        for Z, a in ((2, 0.9), (2, 1.0), (1, 0.9), (1, 1.0)):
            ok, n, sw = agg(Z, a)
            parts.append(f"|Z|={Z}, alpha={a:g}: {ok}/{n} at the best value, median sweeps {min(sw):.0f}-{max(sw):.0f}")
        L.append(f"- Learning curves (start margin {TMAZE_CURVE_MARGIN}, slack {TMAZE_CURVE_SLACK}; best J_phi = 1 with |Z| = 2 "
                 f"and 1/2 with |Z| = 1): " + '; '.join(parts) + '.')
    if 'tmaze_slack' in notes:
        ts = notes['tmaze_slack']
        L.append(f"- Slack in place of memory (|Z| = 1, start margin 1): the memoryless route policy reaches J_phi = 1 on every "
                 f"layout from slack {ts['w_exists']}; the ascent (alpha = 0.9) reaches J_phi = 1 in " +
                 ', '.join(f"W={W}: {v:.2f}" for W, v in zip(ts['W'], ts['pooled'])) + ' of runs (pooled over L); by L: ' +
                 '; '.join(f"L={Lk}: " + ', '.join(f'{v:.2f}' for v in vs) for Lk, vs in ts['by_L'].items()) +
                 f"; lazy alpha = 1: at most {max(ts['lazy']):.2f}.")
    if 'memstart' in notes:
        by_start = defaultdict(list)
        for key, (solved, n, bits) in notes['memstart'].items():
            grp, st = key.split('|')
            by_start[st].append(f"{grp.replace('$', '')}: {solved}/{n}")
        L.append('- (P1), |Z| = 2, runs that store and use the cue, by start: ' +
                 '; '.join(f"{st}: " + ', '.join(v) for st, v in by_start.items()) + '.')
    if 'E1' in notes:
        conv = max(st['conv_max'] for st in notes['E1'])
        L.append(f"- Chain: from the converged (P1) policy, joint greedy with a spare bit returned at most J_r = {conv:.1e}: "
                 f"the spare bit is never used.")

    L += ['', '## Proposition 4: the memory gap can be arbitrarily large', '']
    if 'E1' in notes:
        st = notes['E1']
        L.append(f"- Joint greedy with |Z| = 1 never exceeded (1 - c^(1/T))^T (largest excess "
                 f"{max(x['jg1_over'] for x in st):.1e}); the Lagrangian method returned 1 - c to within "
                 f"{max(x['lag_err'] for x in st):.1e} for every T and c.")
        for c in sorted({x['c'] for x in st}):
            rr = {x['T']: x for x in st if x['c'] == c}
            L.append(f"  - c = {c:g}: with a spare bit from a random start, joint greedy reached v2_bar in "
                     + ', '.join(f"T={t}: {rr[t]['spare_reach']}/{rr[t]['n']}" for t in sorted(rr)) + '.')

    L += ['', '## Equation (9) and Proposition 5: one latched bit; two policies suffice', '']
    if 'E2' in notes:
        for Ln in notes['E2']:
            if Ln['c'] != 0.75:
                continue
            cl = Ln['cell']
            v2s = f"{Ln['v2']:.3g}" if np.isfinite(Ln['v2']) else f">= {Ln['v2_found']:.3g} (best found, no closed form)"
            L.append(f"- Survey maze K={Ln['K']}, s={Ln['s']}, c=0.75: v2 = {v2s}, v2_bar = {Ln['v2_bar']:.3g}; joint greedy "
                     f"|Z|=2 median {cl['joint_Z2']['med']:.3g}; |Z|=4 reached v2_bar in {cl['joint_Z4']['reach']}/"
                     f"{cl['joint_Z4']['n']}; Lagrangian median {cl['lagrangian']['med']:.3g}.")

    L += ['', '## Theorem 6: the Lagrangian method', '']
    L.append(f"- Bound of part (ii) checked on {ch['lbs_checked']} runs where v2_bar and eps are known exactly: held in "
             f"{ch['lbs_held']}; largest v2_bar - J_r(pi_theta) {ch['lbs_gap_max']:.1e}; largest certificate "
             f"{ch['lbs_cert_max']:.1e}. Every mixture met c to within {ch['lag_max_phi_err']:.1e}; reported values matched "
             f"exact evaluation to {ch['lag_curve_check']:.1e}.")
    if 'E2' in notes:
        eps = max(Ln['eps_max'] for Ln in notes['E2'] if np.isfinite(Ln['eps_max']))
        L.append(f"- Survey maze: largest eps at the bracket ends {eps:.1e}.")
    if 'E4' in notes:
        for cfg, R in notes['E4'].items():
            c = R['cells']
            L.append(f"- RockSample {cfg}: joint greedy {c['joint_Z2'][0]:.2f}; Lagrangian (best pair) {c['lagrangian'][0]:+.2f}, "
                     f"polish {c['polish'][0]:+.2f}; same-compute restarts (k={R['k']}) {c['restarts'][0]:+.2f}; |Z|=4 "
                     f"{c['joint_Z4'][0]:+.2f}; exp. gradient average {c['eg_avg'][0]:+.2f}; median sweeps joint "
                     f"{R['sweeps']['joint']:.0f} vs Lagrangian {R['sweeps']['lagrangian']:.0f}.")
    if 'eg_long' in notes:
        order = lambda kv: (float(kv[0].split('|')[0]), kv[0].split('|')[1] != 'bisect', int(kv[0].split('|')[2]))
        for key, (val, phi, lam, n) in sorted(notes['eg_long'].items(), key=order):
            c, kind, K = key.split('|')
            L.append(f"- 3 rocks, c = {c}, {'bisection' if kind == 'bisect' else 'exp. gradient'} with {K} calls: J_r {val:.3f}, "
                     f"J_phi {phi:.3f}, final multiplier {lam:.3g} ({n} runs).")
    thr = load(root, 'threshold')
    if thr:
        g = by(thr, lambda r: r['job']['task'], threshold)
        for (task, c), rs in sorted(g.items()):
            L.append(f"- Threshold sweep, {task}, c = {c:g}: kappa {q([r['rec'].get('kappa') for r in rs], 50):.3g}, bracket "
                     f"width {q([r['rec'].get('width') for r in rs], 50):.3g}, certificate "
                     f"{q([r['rec'].get('certificate') for r in rs], 50):.3g}.")

    f = Path(root) / 'labels' / 'results.json'
    if f.exists():
        rows = [r for r in json.loads(f.read_text(encoding='utf-8')) if r['status'] == 'done' and r['feasible']]
        L += ['', '## Lemma 1 (appendix): reported vs. true satisfaction', '']
        for sensor in ('symmetric', 'optimistic'):
            rs = [r for r in rows if r['sensor'] == sensor and r['margin'] == 'none' and r['eta'] > 0]
            if rs:
                ratio = max(abs(r['J_phi'] - r['J_phi_true']) / r['eta'] for r in rs)
                sign = np.median([np.sign(r['J_phi'] - r['J_phi_true']) for r in rs])
                L.append(f"- {sensor} sensor: |J_phi - J_phi^S| <= {ratio:.2f} eta in every run (report "
                         f"{'overstates' if sign > 0 else 'understates'} the truth); bound (T+1) eta = {rs[0]['T'] + 1} eta.")
        ok = [r for r in rows if r['sensor'] == 'optimistic' and r['margin'] == 'rocks']
        if ok:
            L.append(f"- Optimistic sensor planned at c + (K+1) eta: min true J_phi^S {min(r['J_phi_true'] for r in ok):.3f} "
                     f"(c = {ok[0]['c']:g}), feasible up to eta = {max(r['eta'] for r in ok):g}.")
    (out / 'numbers.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    (out / 'notes.json').write_text(json.dumps(notes, indent=1, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x)), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='runs_paper')
    ap.add_argument('--main', default='runs_main', help='the earlier main-text runs (T-maze (P1), start ablation)')
    ap.add_argument('--out', default='reports_paper')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    notes = {}
    fig3(a.root, out, notes)
    fig4(a.main, out, notes)
    table3_tmaze(a.main, out, notes)
    fig4_tmaze(a.main, out, notes)
    fig_tmaze_slack(a.root, out, notes)
    fig5_rock(a.root, out, notes)
    e4(a.root, out, notes)
    e5(a.root, out, notes)
    e_memstart(a.root, out, notes)
    e_eglong(a.root, out, notes)
    numbers(a.root, out, notes)
    print('wrote', sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
