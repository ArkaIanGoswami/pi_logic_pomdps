"""The experiments grouped by the result of the paper that each one tests."""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import time

RESULTS = [
    ('chain', 'The memory gap on the chain instance of Example 1',
     'chain instance, T = 2..12, c in {1/4, 1/2, 3/4}, 50 seeds: CPI with |Z| = 1 and with a spare bit '
     '(random start; converged (P1) start), and LBS', 0.2),
    ('polar', '(P2) on the polar maze',
     'polar maze, K in {2, 3, 4, 6}, budget s in {0..3}, 50 seeds: CPI with |Z| in {1, 2, 4}, LBS and exponentiated '
     'gradient; random policies for the attainable set; the best value found where v2 has no closed form '
     '(600 starts per maze)', 0.2),
    ('tmaze', '(P1) on the T-maze, the start ablation, and slack in place of memory',
     'cue-key-door T-maze (P1) over corridor length, slack and margin, |Z| in {1, 2}, alpha in {0.9, 1}; start '
     'ablation on the T-maze and on the polar maze (copy, copy + coin, Dirichlet, uniform); slack up to 16 with '
     '|Z| = 1 and start margin 1, with the memoryless route policy checked on every layout; 50 seeds', 1.0),
    ('rocksample', '(P2) on RockSample: CPI, LBS and LBS + CPI against the baselines',
     'RockSample with 3 and 5 rocks at three thresholds: CPI with |Z| in {2, 4}, the Trial-and-Error step, LBS and '
     'exponentiated gradient (16 calls), all 50 seeds; exponentiated gradient with 64 and 256 calls '
     '(3 rocks, 20 seeds); the threshold sweep towards the ceiling', 37.0),
]
ORDER = [r[0] for r in RESULTS]
NOTE_CHECKS = ('The stage identity and the feasibility of the joint update need no runs of their own. '
               'report_paper.py checks them on every joint run of chain, polar and rocksample.')


def run(name, a):
    import paper
    w = ['--workers', str(a.workers)]
    t0 = time.time()
    print(f'\n===== {name} =====', flush=True)
    if name == 'chain':
        paper.main(['--exp', 'chain', '--root', a.root] + w)
    elif name == 'polar':
        paper.main(['--exp', 'tgap,cloud', '--root', a.root] + w)
        import tgap_search
        tgap_search.main(['--n', '300', '--root', a.root])
    elif name == 'tmaze':
        import experiments
        experiments.main(['--exp', 'tmaze_main,start_ablation', '--main-text', '--root', a.main] + w)
        paper.main(['--exp', 'memstart,tmaze_slack', '--root', a.root] + w)
    elif name == 'rocksample':
        paper.main(['--exp', 'rock', '--include', 'rocksample3', '--root', a.root] + w)
        paper.main(['--exp', 'threshold', '--root', a.root] + w)
        paper.main(['--exp', 'eg_long', '--root', a.root] + w)
        paper.main(['--exp', 'rock', '--include', 'rocksample5', '--full', '--root', a.root] + w)
    print(f'===== {name} done in {(time.time() - t0) / 60:.1f} min =====', flush=True)


def report(a):
    import report_paper
    report_paper.main(['--root', a.root, '--main', a.main, '--out', a.out])
    print(f'report written to {a.out}/ (numbers.md lists the in-text numbers by result)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--result', default=None, help=f"comma list of {', '.join(ORDER)}; or 'all'")
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--root', default='runs_paper')
    ap.add_argument('--main', default='runs_main', help='folder for the cue-key-door T-maze runs')
    ap.add_argument('--out', default='reports_paper')
    ap.add_argument('--report', action='store_true', help='only regenerate the figures, tables and numbers')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args(argv)
    if a.list or (a.result is None and not a.report):
        print(f'{NOTE_CHECKS}\n')
        for name, result, what, hours in RESULTS:
            print(f'{name:8s} {result}\n         {what}\n         about {hours:g} CPU hours, '
                  f'{hours / max(1, 0.8 * a.workers) * 60:.0f} min on {a.workers} workers\n')
        total = sum(r[3] for r in RESULTS)
        print(f'all      about {total:.0f} CPU hours, {total / max(1, 0.8 * a.workers):.1f} h on {a.workers} workers')
        return
    if a.result:
        names = ORDER if a.result == 'all' else a.result.split(',')
        unknown = [n for n in names if n not in ORDER]
        if unknown:
            raise SystemExit(f'unknown result {unknown}; choose from {ORDER}')
        for n in names:
            run(n, a)
    report(a)


if __name__ == '__main__':
    main()
