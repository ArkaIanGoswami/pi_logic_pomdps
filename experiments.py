"""Every experiment as a resumable list of jobs, and the parallel runner - Entirely Claude-generated (but verified)"""

import argparse
import json
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import cpi
import learn
import lbs
from product import compile_product
from tasks import level_delta, make_task, presets

LEVELS = ('tight', 'strict', 'mid', 'loose')
TMAZE = [name for name in presets() if name.startswith('tmaze_')]
SEEDED = dict(start='copy', ties='random')
DIRICHLET = dict(start='dirichlet', ties='index')
KAPPA_C = (0.950, 0.955, 0.960, 0.965)
KAPPA_M = (1, 2, 5, 10, 20, 50, 100, 200, 500)
ENV_KEYS = ('bad_rock_penalty', 'slack')


SHORT = dict(alpha='a', bad_rock_penalty='pen', holdout='ho', inner_alpha='ia', lazy='lazy', levelset='lv', margin='mg',
             objective='obj', p2_step='p2', resets='rs', start='st', ties='ti', p2_from='from', polish='pol',
             rule='rule', steps='n', iters='K', eta='eta', samples='N')


def _job(exp, kind, task, seed, variant='logic', Z=2, **opts):
    parts = [task, variant, f'Z{Z}'] + [f'{SHORT.get(k, k)}{v}' for k, v in sorted(opts.items()) if k not in ('levels', 'save')]
    jid = '_'.join(str(p) for p in parts).replace(' ', '') + f'_s{seed:03d}'
    return dict(exp=exp, kind=kind, task=task, variant=variant, Z=Z, seed=seed, id=jid, **opts)


def with_options(job, **opts):
    rest = {k: v for k, v in job.items() if k not in ('exp', 'kind', 'task', 'seed', 'variant', 'Z', 'id')}
    return _job(job['exp'], job['kind'], job['task'], job['seed'], job['variant'], job['Z'], **{**rest, **opts})


def experiments(seeds_for):
    E = {}
    jobs = []
    for name in TMAZE:
        for s in seeds_for('tmaze_main', 'tmaze'):
            for Z in (1, 2):
                for alpha in (0.9, 1.0):
                    jobs.append(_job('tmaze_main', 'cpi', name, s, Z=Z, alpha=alpha, **SEEDED))
                jobs.append(_job('tmaze_main', 'cpi', name, s, variant='free', Z=Z, alpha=0.9, **SEEDED))
    E['tmaze_main'] = jobs

    jobs = []
    for L in (10, 20):
        for slack in (0, 1):
            for m in (0, 1):
                name = f'tmaze_L{L}_s{slack}_m{m}'
                for s in seeds_for('start_ablation', 'tmaze'):
                    for start, ties in (('uniform', 'index'), ('dirichlet', 'index'), ('copy', 'index'), ('copy', 'random')):
                        jobs.append(_job('start_ablation', 'cpi', name, s, Z=2, alpha=0.9, start=start, ties=ties))
                    jobs.append(_job('start_ablation', 'cpi', name, s, Z=1, alpha=0.9, start='uniform', ties='index'))
    for k in (3, 5):
        for s in seeds_for('start_ablation', f'rocksample{k}'):
            for opts in (DIRICHLET, SEEDED):
                jobs.append(_job('start_ablation', 'cpi', f'rocksample{k}_tight', s, alpha=0.9,
                                 levels=('tight', 'strict'), **opts))
    E['start_ablation'] = jobs

    jobs = []
    for k in (3, 5):
        for s in seeds_for('alpha_ablation', f'rocksample{k}'):
            for alpha in (1.0, 0.9):
                jobs.append(_job('alpha_ablation', 'cpi', f'rocksample{k}_tight', s, alpha=alpha, objective='P1', **DIRICHLET))
            jobs.append(_job('alpha_ablation', 'cpi', f'rocksample{k}_tight', s, alpha=1.0, lazy=False, objective='P1',
                             **DIRICHLET))
    E['alpha_ablation'] = jobs

    jobs = []
    for k in (3, 5, 7):
        for s in seeds_for('rocksample_main', f'rocksample{k}'):
            jobs.append(_job('rocksample_main', 'cpi', f'rocksample{k}_tight', s, alpha=0.9, levels=LEVELS, **DIRICHLET))
            jobs.append(_job('rocksample_main', 'cpi', f'rocksample{k}_tight', s, variant='free', alpha=0.9, **DIRICHLET))
            if k == 3:
                jobs.append(_job('rocksample_main', 'cpi', f'rocksample{k}_tight', s, Z=1, alpha=0.9, levels=LEVELS, **DIRICHLET))
    E['rocksample_main'] = jobs

    jobs = []
    for k in (3, 5):
        for s in seeds_for('penalty_ablation', f'rocksample{k}'):
            jobs.append(_job('penalty_ablation', 'cpi', f'rocksample{k}_tight', s, alpha=0.9, levels=LEVELS,
                             bad_rock_penalty=0.0, **DIRICHLET))
            jobs.append(_job('penalty_ablation', 'cpi', f'rocksample{k}_tight', s, variant='free', alpha=0.9,
                             bad_rock_penalty=0.0, **DIRICHLET))
    E['penalty_ablation'] = jobs

    jobs = []
    for k in (3, 5, 7):
        for s in seeds_for('naive_step', f'rocksample{k}'):
            jobs.append(_job('naive_step', 'cpi', f'rocksample{k}_tight', s, alpha=0.9, levels=LEVELS, p2_step='naive',
                             **DIRICHLET))
    E['naive_step'] = jobs

    jobs = []
    for s in seeds_for('lagrangian', 'twostep'):
        for inner in (1.0, 0.9):
            jobs.append(_job('lagrangian', 'lagrangian', 'twostep', s, Z=1, alpha=0.9, inner_alpha=inner, **DIRICHLET))
    for k in (3, 5, 7):
        for s in seeds_for('lagrangian', f'rocksample{k}'):
            for inner in ((1.0, 0.9) if k < 7 else (0.9,)):
                jobs.append(_job('lagrangian', 'lagrangian', f'rocksample{k}_tight', s, alpha=0.9, inner_alpha=inner, **DIRICHLET))
    E['lagrangian'] = jobs

    jobs = []
    for name in TMAZE:
        for s in seeds_for('ce_main', 'tmaze'):
            for Z in (1, 2):
                jobs.append(_job('ce_main', 'ce', name, s, Z=Z, alpha=0.9, m=1, **SEEDED))
    for k in (3, 5, 7):
        for s in seeds_for('ce_main', f'rocksample{k}'):
            for m in ((1, 3, 10, 30, 100) if k < 7 else (10, 100)):
                jobs.append(_job('ce_main', 'ce', f'rocksample{k}_tight', s, alpha=0.9, m=m, levels=LEVELS, **DIRICHLET))
                jobs.append(_job('ce_main', 'ce', f'rocksample{k}_tight', s, variant='free', alpha=0.9, m=m, **DIRICHLET))
    E['ce_main'] = jobs

    jobs = []
    for s in seeds_for('kappa', 'rocksample5'):
        jobs.append(_job('kappa', 'cpi', 'rocksample5_tight', s, alpha=0.9, levels=KAPPA_C, **DIRICHLET))
        for m in KAPPA_M:
            jobs.append(_job('kappa', 'ce', 'rocksample5_tight', s, alpha=0.9, m=m, levels=KAPPA_C, margin='half',
                             resets=2000 * m, holdout=50000, save=False, **DIRICHLET))
    E['kappa'] = jobs
    return E


def main_text(E):
    def ce(j):
        return (j['task'].startswith('rocksample') and j['variant'] == 'logic'
                and not (j['task'].startswith('rocksample7') and j['seed'] >= 10))
    return dict(tmaze_main=E['tmaze_main'],
                start_ablation=[j for j in E['start_ablation'] if j['task'].startswith('tmaze')],
                rocksample_main=E['rocksample_main'],
                naive_step=E['naive_step'],
                lagrangian=[j for j in E['lagrangian'] if j['task'] == 'twostep'],
                ce_main=[with_options(j, levels=('strict',), levelset='strict', margin='half') for j in E['ce_main'] if ce(j)],
                kappa=E['kappa'])


FULL_SEEDS = dict(tmaze=50, twostep=20, rocksample3=50, rocksample5=50, rocksample7=20)
ABLATION_SEEDS = dict(tmaze=50, twostep=20, rocksample3=20, rocksample5=20, rocksample7=10)


def default_seeds(exp, family):
    if exp == 'kappa':
        return range(20)
    if exp in ('start_ablation', 'alpha_ablation', 'penalty_ablation', 'lagrangian'):
        return range(ABLATION_SEEDS[family])
    return range(FULL_SEEDS[family])


def pilot_seeds(exp, family):
    if family == 'rocksample7':
        return range({'rocksample_main': 3}.get(exp, 1))
    return range(10)


_MODELS = {}


def model_for(task):
    if task['family'] == 'tmaze':
        return compile_product(task)
    key = json.dumps({k: v for k, v in task['config'].items() if k != 'seed'}, sort_keys=True, default=str)
    if key not in _MODELS:
        _MODELS[key] = compile_product(task)
    return _MODELS[key]


def thresholds(task, job):
    out = []
    if task['objective'] != 'P2':
        return out
    v1 = task['env']['info'].get('ceiling', 1.0)
    for lv in job.get('levels', ()):
        c = 1.0 - level_delta(task['env'], lv) if isinstance(lv, str) else float(lv)
        plan = c + (v1 - c) / 2 if job.get('margin') == 'half' else c + job.get('kappa', 0.0)
        out.append((str(lv), c, plan))
    return out


def describe(task, job):
    d = dict(exp=job['exp'], kind=job['kind'], task=job['task'], family=task['family'], variant=job['variant'],
             objective=task['objective'], Z=job['Z'], seed=job['seed'], alpha=job.get('alpha', 0.9),
             lazy=job.get('lazy', job.get('alpha', 0.9) == 1.0), p2_step=job.get('p2_step', 'joint'),
             start=job.get('start', 'dirichlet'), ties=job.get('ties', 'index'), m=job.get('m'),
             inner_alpha=job.get('inner_alpha'), horizon=task['horizon'], gamma=task['gamma'])
    if task['family'] == 'tmaze':
        cfg = task['env']['config']
        d.update(L=cfg['corridor_length'], slack=task['env']['info']['slack'], margin=cfg['start_margin'])
    if task['family'] == 'rocksample':
        d.update(rocks=len(task['env']['layout']['rocks']), ceiling=task['env']['info']['ceiling'],
                 bad_rock_penalty=task['env']['config']['bad_rock_penalty'])
    if task['family'] == 'chain':
        d.update(decisions=task['env']['info']['decisions'])
    if task['family'] == 'tmaze_gap':
        d.update(surveys=task['env']['info']['surveys'], budget=task['env']['info']['budget'],
                 L=task['env']['info']['corridor_length'])
    for k in ('p2_from', 'rule', 'steps', 'iters', 'eta', 'polish'):
        if k in job:
            d[k] = job[k]
    return d


def monotone(log, tol=1e-9):
    for a, b in zip(log, log[1:]):
        key = 'J_r' if b['phase'] in ('P2', 'P0') else 'J_phi'
        if b[key] < a[key] - tol * (1 + abs(a[key])):
            return False
    return True


def max_drop(log):
    d = 0.0
    for a, b in zip(log, log[1:]):
        if b['phase'] == 'init':
            continue
        key = 'J_r' if b['phase'] in ('P2', 'P0') else 'J_phi'
        d = max(d, a[key] - b[key])
    return d


def summary(log):
    p = [l for l in log if l['phase'] not in ('init',)]
    return dict(sweeps=len(p), p1_sweeps=sum(l['phase'] == 'P1' for l in p), p2_sweeps=sum(l['phase'] == 'P2' for l in p),
                identity_err=max([l['identity_err'] for l in p] or [0.0]),
                min_phi_P2=min([l['min_phi'] for l in p if l['phase'] == 'P2'] or [np.nan]),
                dual_gap=max([l['gap'] for l in p if l['phase'] == 'P2'] or [0.0]),
                rejected=sum(l.get('rejected', 0) for l in p),
                lam_pos_frac=(sum(l['lam_pos'] for l in p if l['phase'] == 'P2') /
                              max(1, sum(l['stages'] for l in p if l['phase'] == 'P2'))),
                lam_max=max([l['lam_max'] for l in p if l['phase'] == 'P2'] or [0.0]),
                short_frac=(sum(l.get('short', 0) for l in p if l['phase'] == 'P2') /
                            max(1, sum(l['stages'] for l in p if l['phase'] == 'P2'))),
                monotone=monotone(log), max_drop=max_drop(log),
                secs=sum(l.get('secs', 0) for l in p))


def run_cpi(task, model, job, out):
    cfg = dict(alpha=job.get('alpha', 0.9), lazy=job.get('lazy', job.get('alpha', 0.9) == 1.0),
               start=job.get('start', 'dirichlet'), ties=job.get('ties', 'index'), p2_step=job.get('p2_step', 'joint'),
               p2_from=job.get('p2_from'))
    levels = [(n, c) for n, c, _ in thresholds(task, job)]
    res = cpi.solve(task, model, cfg, levels, rng=np.random.default_rng(job['seed']))
    records, logs = [], {}
    for name, r in res.items():
        rec = dict(level=name, c=r.get('c'), feasible_start=r['feasible_start'], **summary(r['log']))
        J = cpi.evaluate(task, model, r['pol'])
        rec.update(J_phi=J['J_phi'], J_r=J['J_r'], J_reject=J['J_reject'],
                   feasible=None if r.get('c') is None else bool(J['J_phi'] >= r['c'] - 1e-9))
        if task['family'] in ('tmaze', 'tmaze_gap') and job['Z'] > 1:
            rec['memory_bits'], rec['memory_peak_t'] = cpi.memory_information(task, model, r['pol'], lambda s: s[0])
        if job['variant'] == 'free':
            logic = make_task(job['task'], variant='logic', seed=job['seed'], memory_size=job['Z'], **env_overrides(job))
            lm = model_for(logic)
            pol = cpi.transfer(r['pol'], model['rows'], lm['rows'], cpi.initial_policy(logic, lm, cfg['start'], job['seed']),
                               key=lambda row: row[0])
            Jl = cpi.evaluate(logic, lm, pol)
            rec.update(logic_J_phi=Jl['J_phi'], logic_J_r=Jl['J_r'])
        rec['peak_mb'] = _peak_mb()
        records.append(rec)
        logs[name] = r['log']
        if job.get('save', True):
            cpi.save_policy(out / f'policy_{name}.npz', model, r['pol'], task)
    return records, logs


def run_ce(task, model, job, out):
    cfg = dict(m=job['m'], alpha=job.get('alpha', 0.9), lazy=job.get('alpha', 0.9) == 1.0,
               start=job.get('start', 'dirichlet'), ties=job.get('ties', 'index'),
               resets=job.get('resets', 2000), holdout=job.get('holdout', 5000), beta=0.05,
               track_true=task['family'] != 'rocksample' or len(task['env']['layout']['rocks']) < 7)
    records, pols, _ = learn.run(task, cfg, thresholds(task, job), job['seed'], true_model=model)
    logs = {}
    for rec in records:
        pol, log = pols[rec['level']]
        rec.update(summary(log))
        logs[rec['level']] = log
        if job.get('save', True):
            cpi.save_policy(out / f'policy_{rec["level"]}.npz', model, pol, task)
    return records, logs


def run_lagrangian(task, model, job, out):
    c = 1.0 - task['delta']
    cfg = dict(alpha=0.9, start='dirichlet', ties='index', inner_alpha=job['inner_alpha'])
    p1 = cpi.solve(dict(task, objective='P1'), model, cfg)['P1']
    joint = cpi.solve(task, model, cfg, [('c', c)])['c']
    start = cpi.initial_policy(task, model, 'dirichlet', job['seed'])
    t0 = time.time()
    lag = cpi.lagrangian(task, model, c, p1['pol'], start, cfg)
    rec = dict(level='tight', c=c, joint_J_phi=joint['J']['phi'], joint_J_r=joint['J']['r'],
               p1_J_phi=p1['J']['phi'], status=lag['status'], J_phi=lag.get('J_phi'), J_r=lag.get('J_r'),
               theta=lag.get('theta'), lam_lo=lag.get('lam_lo'), lam_hi=lag.get('lam_hi'), width=lag.get('width'),
               kappa_warm=lag.get('kappa'), J_r_formula=lag.get('J_r_formula'), lag_secs=time.time() - t0,
               ascents=len(lag.get('log', [])))
    if task['family'] == 'twostep':
        rec.update(v2=0.25, v2_bar=0.75)
    if lag['status'] == 'mixed' and job.get('save', True):
        cpi.save_policy(out / 'policy_mixture.npz', model, lag['mixture'], lag['mixture_task'], theta=lag['theta'])
    return [rec], dict(bracket=lag.get('bracket', []), ascents=lag.get('log', []), joint=joint['log'])


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items() if not isinstance(v, np.ndarray)}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _peak_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        return None


LBS_KEYS = ('status', 'c', 'kappa', 'R', 'theta', 'lam_lo', 'lam_hi', 'width', 'certificate', 'bracket_J_phi',
              'bracket_J_r', 'J_phi', 'J_r', 'pair_value', 'pair_theta', 'pair_tags', 'pair_points', 'polish_J_phi',
              'polish_J_r', 'polish_sweeps', 'eps_lo', 'eps_hi', 'J_hi', 'J_lo', 'sweeps', 'p1_sweeps', 'warm_J_phi',
              'warm_J_r', 'secs', 'secs_with_polish', 'points', 'curve_check', 'rule', 'iters', 'B', 'eta',
              'avg_J_phi', 'avg_J_r', 'avg_violation', 'avg_policies', 'avg_bits', 'lams', 'iter_J_phi', 'iter_J_r')


def _multiplier_job(task, model, job, out, method):
    c = float(job['c'])
    start = cpi.initial_policy(task, model, job.get('start', 'dirichlet'), job['seed'])
    cfg = dict(alpha=job.get('alpha', 0.9), inner_alpha=job.get('inner_alpha', 0.9), ties=job.get('ties', 'index'),
               seed=job['seed'], max_sweeps=job.get('max_sweeps', 60))
    if method == 'bisect':
        res = lbs.bisection(task, model, c, start, cfg, steps=job.get('steps', 14), rule=job.get('rule', 'mid'),
                                polish=job.get('polish', True))
    else:
        res = lbs.exponentiated_gradient(task, model, c, start, cfg, iters=job.get('iters', 16), eta=job.get('eta'))
    rec = dict(level=f'{c:g}', method=method, **{k: _plain(res[k]) for k in LBS_KEYS if k in res})
    rec['feasible'] = bool(rec.get('J_phi') is not None and rec['J_phi'] >= c - 1e-9)
    rec['peak_mb'] = _peak_mb()
    logs = dict(curve=_plain(res.get('curve', [])), avg_curve=_plain(res.get('avg_curve', [])),
                bracket=_plain(res.get('bracket', [])), oracle=_plain(res.get('log', [])),
                points=_plain(res['incumbent'].points) if 'incumbent' in res else [])
    if job.get('save', False) and 'policy' in res:
        cpi.save_policy(out / f'policy_{method}.npz', model, res['policy'], res['policy_task'])
    return [rec], logs


def run_bisect(task, model, job, out):
    return _multiplier_job(task, model, job, out, 'bisect')


def run_eg(task, model, job, out):
    return _multiplier_job(task, model, job, out, 'eg')


def run_route(task, model, job, out):
    import tmaze_route
    v, status = tmaze_route.route_value(task, model)
    return [dict(level='route', J_phi=v, route_status=status, route_exists=bool(v is not None and v >= 1 - 1e-9))], {}


def run_cloud(task, model, job, out):
    rng = np.random.default_rng(job['seed'])
    H, R, Z, A = task['horizon'], len(model['rows']), task['memory_size'], model['A']
    legal = model['legal']
    pts = []
    for k in range(job.get('samples', 2000)):
        conc = rng.choice((0.1, 0.3, 1.0))
        pol = np.zeros((H, R, Z, A, Z))
        if k % 2 == 0:
            for t in range(H):
                for r in range(R):
                    for z in range(Z):
                        w = rng.dirichlet(np.full(legal[r].sum() * Z, conc))
                        pol[t, r, z, legal[r], :] = w.reshape(-1, Z)
        else:
            for t in range(H):
                for r in range(R):
                    y, q = model['rows'][r]
                    for z in range(Z):
                        act = np.zeros(A)
                        act[legal[r]] = rng.dirichlet(np.full(legal[r].sum(), conc))
                        if t == 0 and isinstance(y[1], str) and y[1].startswith('cue'):
                            pol[t, r, z, :, int(y[1][3:]) % Z] = act
                        else:
                            pol[t, r, z, :, z] = act
        J = cpi.evaluate(task, model, pol)
        pts.append((J['J_phi'], J['J_r']))
    return [dict(level='cloud', samples=len(pts))], dict(points=pts)


def env_overrides(job):
    return {k: job[k] for k in ENV_KEYS if k in job}


def run_job(args):
    job, root = args
    out = Path(root) / job['exp'] / job['id']
    if (out / 'result.json').exists():
        return job['id'], 'skipped', 0.0
    t0 = time.time()
    try:
        task = make_task(job['task'], variant=job['variant'], seed=job['seed'], memory_size=job['Z'],
                         objective=job.get('objective'), **env_overrides(job))
        model = model_for(task)
        out.mkdir(parents=True, exist_ok=True)
        records, logs = {'cpi': run_cpi, 'ce': run_ce, 'lagrangian': run_lagrangian, 'bisect': run_bisect, 'eg': run_eg,
                         'cloud': run_cloud, 'route': run_route}[job['kind']](task, model, job, out)
        meta = describe(task, job)
        for rec in records:
            for k, v in meta.items():
                rec.setdefault(k, v)
        (out / 'log.json').write_text(json.dumps(logs, default=float), encoding='utf-8')
        (out / 'result.json').write_text(json.dumps(dict(job=job, records=records, secs=time.time() - t0),
                                                    indent=1, default=float), encoding='utf-8')
        return job['id'], 'done', time.time() - t0
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        return job['id'], 'error', time.time() - t0


COST = dict(rocksample7=100, rocksample5=10, rocksample3=2, tmaze=1, twostep=1, chain=1, tgap=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', default='all', help="experiment name, comma list, or 'all'")
    ap.add_argument('--seeds', default=None, help='a:b overrides every seed budget')
    ap.add_argument('--pilot', action='store_true', help='10 seeds (fewer for 7 rocks)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--root', default='runs')
    ap.add_argument('--include', default='', help='only tasks starting with these prefixes (comma list)')
    ap.add_argument('--exclude', default='', help='skip tasks starting with these prefixes (comma list)')
    ap.add_argument('--bad-rock-penalty', type=float, default=None,
                    help='RockSample reward for sampling a bad rock, in every experiment (default: the environment value, -10); '
                         'use a separate --root for each value')
    ap.add_argument('--main-text', action='store_true',
                    help='only the four main-text experiments, reduced settings, RockSample without the bad-rock penalty '
                         '(unless --bad-rock-penalty is given); results go to runs_main unless --root is given')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.seeds:
        lo, _, hi = a.seeds.partition(':')
        fixed = range(int(lo), int(hi)) if hi else [int(lo)]
        seeds_for = lambda exp, fam: fixed
    else:
        seeds_for = pilot_seeds if a.pilot else default_seeds
    E = experiments(seeds_for)
    if a.main_text:
        E = main_text(E)
        a.bad_rock_penalty = 0.0 if a.bad_rock_penalty is None else a.bad_rock_penalty
        a.root = 'runs_main' if a.root == 'runs' else a.root
    if a.bad_rock_penalty is not None:
        E = {n: [with_options(j, bad_rock_penalty=a.bad_rock_penalty)
                 if j['task'].startswith('rocksample') and 'bad_rock_penalty' not in j else j for j in js] for n, js in E.items()}
    names = list(E) if a.exp == 'all' else a.exp.split(',')
    if a.list:
        for n in E:
            print(f'{n:16s} {len(E[n]):6d} jobs')
        return
    jobs = [j for n in names for j in E[n]]
    inc, exc = [x for x in a.include.split(',') if x], [x for x in a.exclude.split(',') if x]
    jobs = [j for j in jobs if (not inc or j['task'].startswith(tuple(inc))) and not j['task'].startswith(tuple(exc) or ('\0',))]
    jobs.sort(key=lambda j: -COST.get(j['task'].split('_')[0], 1))
    print(f'{len(jobs)} jobs, {a.workers} workers, root {a.root}', flush=True)
    t0, done = time.time(), 0
    args = [(j, a.root) for j in jobs]
    pool = Pool(a.workers) if a.workers > 1 else None
    results = pool.imap_unordered(run_job, args, chunksize=1) if pool else map(run_job, args)
    try:
        for jid, status, secs in results:
            done += 1
            if status == 'error' or done % 50 == 0 or secs > 30:
                print(f'[{done}/{len(jobs)} {time.time() - t0:7.0f}s] {status:7s} {secs:6.1f}s {jid}', flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    print(f'finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
