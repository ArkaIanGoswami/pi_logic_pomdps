"""One method on one task and one seed."""

import argparse
import json
from pathlib import Path
from experiments import _job, run_job, with_options


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kind', default='cpi', choices=('cpi', 'ce', 'lagrangian'))
    ap.add_argument('--task', required=True)
    ap.add_argument('--variant', default='logic', choices=('logic', 'free'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--memory-size', type=int, default=2)
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--start', default=None, choices=('dirichlet', 'uniform', 'copy'),
                    help='default: copy for the T-maze (the seeded start), dirichlet otherwise, as in experiments.py')
    ap.add_argument('--ties', default=None, choices=('index', 'random'), help='default: random for the T-maze, index otherwise')
    ap.add_argument('--levels', nargs='*', default=['tight'], help="P2 levels: tight strict mid loose, or thresholds c")
    ap.add_argument('--m', type=int, default=10, help='queries per (state, action) for --kind ce')
    ap.add_argument('--inner-alpha', type=float, default=1.0)
    ap.add_argument('--bad-rock-penalty', type=float, default=None, help='RockSample only (default: the environment value, -10)')
    ap.add_argument('--root', default='runs')
    a = ap.parse_args(argv)
    maze = a.task.startswith('tmaze')
    a.start = a.start or ('copy' if maze else 'dirichlet')
    a.ties = a.ties or ('random' if maze else 'index')
    levels = tuple(l if l.isalpha() else float(l) for l in a.levels)
    opts = dict(alpha=a.alpha, start=a.start, ties=a.ties)
    if a.kind == 'cpi' or a.kind == 'ce':
        opts['levels'] = levels
    if a.kind == 'ce':
        opts['m'] = a.m
    if a.kind == 'lagrangian':
        opts['inner_alpha'] = a.inner_alpha
    job = _job('single', a.kind, a.task, a.seed, variant=a.variant, Z=a.memory_size, **opts)
    if a.bad_rock_penalty is not None:
        job = with_options(job, bad_rock_penalty=a.bad_rock_penalty)
    jid, status, secs = run_job((job, a.root))
    out = Path(a.root) / 'single' / jid
    print(status, f'{secs:.1f}s', out)
    if (out / 'result.json').exists():
        for r in json.loads((out / 'result.json').read_text(encoding='utf-8'))['records']:
            print({k: r[k] for k in ('level', 'J_phi', 'J_r', 'feasible', 'certified') if k in r})
    elif (out / 'error.txt').exists():
        print((out / 'error.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
