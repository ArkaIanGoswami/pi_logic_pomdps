"""The best (P2) value found on polar mazes where v_2(c) has no closed form."""

import sys
import numpy as np
import cpi
from tasks import make_task
from experiments import model_for


def best_found(name, c, n, starts=('copycoin', 'copy'), seed0=10_000):
    task = make_task(name, seed=0, memory_size=2, delta=1 - c)
    model = model_for(task)
    best = {}
    for start in starts:
        b = -np.inf
        for seed in range(n):
            res = cpi.solve(task, model, dict(alpha=0.9, start=start, ties='random', p2_step='joint'), [('c', c)],
                            rng=np.random.default_rng(seed0 + seed))['c']
            J = cpi.evaluate(task, model, res['pol'])
            if J['J_phi'] >= c - 1e-9:
                b = max(b, J['J_r'])
        best[start] = b
    return max(best.values()), best


def main(argv=None):
    import argparse, json
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=300, help='starts per kind of start')
    ap.add_argument('--root', default='runs_paper')
    a = ap.parse_args(argv)
    out = []
    for name, c in (('tgap_K4_s1', 0.75), ('tgap_K4_s2', 0.75)):
        best, by = best_found(name, c, a.n)
        out.append(dict(task=name, K=int(name.split('_')[1][1:]), s=int(name.split('_')[2][1:]), c=c, n=a.n, best=best, by=by))
        print(name, c, best, by, flush=True)
    path = Path(a.root) / 'tgap_search' / 'results.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
