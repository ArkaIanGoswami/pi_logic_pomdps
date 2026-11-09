"""The job lists behind the tables and figures of the paper - Entirely Claude-generated (but verified)"""

import argparse
import time
from multiprocessing import Pool
from experiments import _job, run_job, with_options

ROOT = 'runs_paper'
SEEDS = 50
ROCK_EXPENSIVE_SEEDS = 20
CHAIN_T = tuple(range(2, 13))
CHAIN_C = (0.25, 0.5, 0.75)
TGAP_TASKS = ('tgap_K2_s0', 'tgap_K3_s0', 'tgap_K4_s0', 'tgap_K6_s0', 'tgap_K4_s1', 'tgap_K4_s2', 'tgap_K4_s3')
TGAP_C = (0.75, 0.6, 0.9)
ROCK_C = {3: (0.55, 0.8, 0.86), 5: (0.55, 0.9, 0.96)}
THRESHOLD = {'rocksample3_tight': (0.8, 0.84, 0.86, 0.87, 0.874), 'tgap_K3_s0': (0.6, 0.75, 0.9, 0.95, 0.99)}
MEMSTART_TASKS = ('tgap_K2_s0', 'tgap_K3_s0', 'tgap_K4_s0', 'tgap_K6_s0') + tuple(
    f'tmaze_L{L}_s{sl}_m{m}' for L in (10, 20) for sl in (0, 1) for m in (0, 1))
MEMSTARTS = (('copy', 'random'), ('copycoin', 'random'), ('dirichlet', 'index'), ('uniform', 'index'))
TMAZE_SLACK_L = (3, 5, 10, 20)
TMAZE_SLACK_W = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16)
EG_LONG_ITERS = (64, 256)
EG_LONG_SEEDS = 20
DIRICHLET = dict(start='dirichlet', ties='index')
SEEDED = dict(start='copycoin', ties='random')
MULTIPLIER_OPTS = dict(alpha=0.9, inner_alpha=0.9, save=False)
KIND_COST = dict(bisect=15, eg=15, cpi=1, cloud=5)
TASK_COST = dict(rocksample5=20, rocksample3=2)


def experiments(full=False):
    E = {}
    jobs = []
    for T in CHAIN_T:
        for c in CHAIN_C:
            lv = dict(levels=(c,), levelset=f'c{c}', save=False)
            for s in range(SEEDS):
                task = f'chain{T}'
                jobs.append(_job('chain', 'cpi', task, s, Z=1, alpha=0.9, **DIRICHLET, **lv))
                jobs.append(_job('chain', 'cpi', task, s, Z=2, alpha=0.9, **DIRICHLET, **lv))
                jobs.append(_job('chain', 'cpi', task, s, Z=2, alpha=1.0, lazy=True, p2_from='converged', **DIRICHLET, **lv))
                jobs.append(_job('chain', 'bisect', task, s, Z=1, c=c, **DIRICHLET, **dict(MULTIPLIER_OPTS, inner_alpha=1.0)))
    E['chain'] = jobs

    jobs = []
    for task in TGAP_TASKS:
        for c in TGAP_C:
            if c != 0.75 and task not in ('tgap_K2_s0', 'tgap_K4_s0'):
                continue
            lv = dict(levels=(c,), levelset=f'c{c}', save=False)
            for s in range(SEEDS):
                for Z in (1, 2, 4):
                    jobs.append(_job('tgap', 'cpi', task, s, Z=Z, alpha=0.9, **SEEDED, **lv))
                jobs.append(_job('tgap', 'bisect', task, s, Z=2, c=c, **SEEDED, **MULTIPLIER_OPTS))
                if c == 0.75:
                    jobs.append(_job('tgap', 'eg', task, s, Z=2, c=c, iters=16, **SEEDED, **MULTIPLIER_OPTS))
    E['tgap'] = jobs

    E['cloud'] = [_job('cloud', 'cloud', 'tgap_K2_s0', s, Z=2, samples=3000, save=False) for s in range(2)]

    jobs = []
    for k, cs in ROCK_C.items():
        task = f'rocksample{k}_tight'
        lv = dict(levels=cs, levelset='paper', save=False, bad_rock_penalty=0.0)
        for s in range(SEEDS):
            jobs.append(_job('rock', 'cpi', task, s, Z=2, alpha=0.9, **DIRICHLET, **lv))
            jobs.append(_job('rock', 'cpi', task, s, Z=4, alpha=0.9, **DIRICHLET, **lv))
            jobs.append(_job('rock', 'cpi', task, s, Z=2, alpha=0.9, p2_step='naive', **DIRICHLET, **lv))
        n_exp = SEEDS if (k == 3 or full) else ROCK_EXPENSIVE_SEEDS
        for s in range(n_exp):
            for c in cs:
                jobs.append(_job('rock', 'bisect', task, s, Z=2, c=c, bad_rock_penalty=0.0, **DIRICHLET, **MULTIPLIER_OPTS))
                jobs.append(_job('rock', 'eg', task, s, Z=2, c=c, iters=16, bad_rock_penalty=0.0, **DIRICHLET, **MULTIPLIER_OPTS))
    E['rock'] = jobs

    jobs = []
    for task, cs in THRESHOLD.items():
        opts = dict(bad_rock_penalty=0.0, **DIRICHLET) if task.startswith('rock') else dict(SEEDED)
        for c in cs:
            for s in range(20):
                jobs.append(_job('threshold', 'bisect', task, s, Z=2, c=c, **opts, **MULTIPLIER_OPTS))
    E['threshold'] = jobs

    E['memstart'] = [_job('memstart', 'cpi', task, s, Z=2, alpha=0.9, objective='P1', start=st, ties=ti, save=False)
                     for task in MEMSTART_TASKS for st, ti in MEMSTARTS for s in range(SEEDS)]

    jobs = []
    for L in TMAZE_SLACK_L:
        for W in TMAZE_SLACK_W:
            task = f'tmaze_L{L}_s0_m1'
            for s in range(SEEDS):
                for alpha in (0.9, 1.0):
                    jobs.append(_job('tmaze_slack', 'cpi', task, s, Z=1, alpha=alpha, slack=W, start='copy',
                                     ties='random', save=False))
                jobs.append(_job('tmaze_slack', 'route', task, s, Z=1, slack=W))
    E['tmaze_slack'] = jobs

    E['eg_long'] = [_job('eg_long', 'eg', 'rocksample3_tight', s, Z=2, c=c, iters=K, bad_rock_penalty=0.0, **DIRICHLET, **MULTIPLIER_OPTS)
                    for K in EG_LONG_ITERS for c in ROCK_C[3] for s in range(EG_LONG_SEEDS)]
    return E


def cost(job):
    calls = job.get('iters', 16) / 16 if job['kind'] == 'eg' else 1
    return KIND_COST.get(job['kind'], 1) * TASK_COST.get(job['task'].split('_')[0], 1) * (2 if job['Z'] == 4 else 1) * calls


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exp', default='all', help="comma list of: chain, tgap, cloud, rock, threshold, memstart, tmaze_slack, eg_long; or 'all'")
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--include', default='', help='only tasks starting with these prefixes (comma list)')
    ap.add_argument('--exclude', default='', help='skip tasks starting with these prefixes (comma list)')
    ap.add_argument('--kinds', default='', help='only these job kinds (comma list of cpi, bisect, eg, cloud)')
    ap.add_argument('--seeds', default=None, help='a:b keeps only seeds in [a, b)')
    ap.add_argument('--full', action='store_true', help=f'50 seeds for the expensive 5-rock arms (default {ROCK_EXPENSIVE_SEEDS})')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    E = experiments(full=a.full)
    names = list(E) if a.exp == 'all' else a.exp.split(',')
    jobs = [j for n in names for j in E[n]]
    inc, exc = [x for x in a.include.split(',') if x], [x for x in a.exclude.split(',') if x]
    kinds = [x for x in a.kinds.split(',') if x]
    jobs = [j for j in jobs if (not inc or j['task'].startswith(tuple(inc)))
            and not (exc and j['task'].startswith(tuple(exc))) and (not kinds or j['kind'] in kinds)]
    if a.seeds:
        lo, _, hi = a.seeds.partition(':')
        jobs = [j for j in jobs if int(lo) <= j['seed'] < (int(hi) if hi else int(lo) + 1)]
    if a.list:
        for n in names:
            js = [j for j in jobs if j['exp'] == n]
            by = {}
            for j in js:
                by[j['kind']] = by.get(j['kind'], 0) + 1
            print(f'{n:10s} {len(js):6d} jobs  {by}')
        return
    jobs.sort(key=lambda j: -cost(j))
    print(f'{len(jobs)} jobs, {a.workers} workers, root {a.root}', flush=True)
    t0, done = time.time(), 0
    args = [(j, a.root) for j in jobs]
    pool = Pool(a.workers) if a.workers > 1 else None
    results = pool.imap_unordered(run_job, args, chunksize=1) if pool else map(run_job, args)
    try:
        for jid, status, secs in results:
            done += 1
            if status == 'error' or done % 100 == 0 or secs > 60:
                print(f'[{done}/{len(jobs)} {time.time() - t0:7.0f}s] {status:7s} {secs:7.1f}s {jid}', flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    print(f'finished in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
