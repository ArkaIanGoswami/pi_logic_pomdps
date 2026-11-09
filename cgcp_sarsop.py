"""CGCP (Walraven and Spaan, 2018, Algorithm 5) with SARSOP solving the subproblems.

    python cgcp_sarsop.py --help
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import cgcp_belief as CB
import lbs

SARSOP = os.environ.get('SARSOP', str(Path(__file__).parent / 'sarsop' / 'src' / 'pomdpsol'))
GAMMA_S = 0.99
FAST = os.environ.get('SARSOP_FAST', '1') != '0'


class Export:
    def __init__(self, task, model, bm):
        self.task, self.model, self.bm = task, model, bm
        H, n = task['horizon'], model['n']
        P = [p.tocsr() for p in model['P']]
        self.legal_node = model['legal'][model['row_of_node']]
        first_legal = self.legal_node.argmax(1)
        self.Aeff = lambda a: np.where(self.legal_node[:, a], a, first_legal)
        reach = [np.flatnonzero(model['initial'] > 0)]
        anyP = sum(P).tocsr()
        for t in range(H):
            v = np.zeros(n)
            v[reach[-1]] = 1.0
            reach.append(np.flatnonzero(anyP.T @ v > 0))
        self.reach = reach
        self.sid = {}
        k = 1
        for t in range(H + 1):
            for j in reach[t]:
                self.sid[(t, int(j))] = k
                k += 1
        self.end, self.N = k, k + 1
        self.P, self.H, self.n, self.A = P, H, n, model['A']
        self._cache = {}

    def _static(self, fast):
        if fast in self._cache:
            return self._cache[fast]
        H, A, g, gs = self.H, self.A, self.task['gamma'], GAMMA_S
        each = (lambda line: [line.replace('{a}', str(a)) for a in range(A)]) if fast else \
            (lambda line: [line.replace('{a}', '*')])
        rows = self.model['row_of_node']
        nobs = len(self.model['rows']) + 1
        out = [f'discount: {gs}', 'values: reward', f'states: {self.N}', f'actions: {A}', f'observations: {nobs}']
        start = np.zeros(self.N)
        start[0] = 1.0
        out.append('start: ' + ' '.join('1' if x else '0' for x in start))
        none = nobs - 1
        T, O, R, acc = [], [], [], []
        init = self.model['initial']
        for j in self.reach[0]:
            T += each(f'T: {{a}} : 0 : {self.sid[(0, int(j))]} {init[j]:.17g}')
        for t in range(H + 1):
            for j in self.reach[t]:
                s = self.sid[(t, int(j))]
                O += each(f'O: {{a}} : {s} : {rows[j]} 1')
                if t == H:
                    T += each(f'T: {{a}} : {s} : {self.end} 1')
                    if self.model['accepting'][j]:
                        acc.append(s)
                    continue
                for a in range(A):
                    ae = int(self.Aeff(a)[j])
                    row = self.P[ae].getrow(j)
                    for jj, p in zip(row.indices, row.data):
                        T.append(f'T: {a} : {s} : {self.sid[(t + 1, int(jj))]} {p:.17g}')
                    r = self.model['R'][j, ae]
                    if r:
                        R.append(f'R: {a} : {s} : * : * {g ** t * r / gs ** (t + 1):.17g}')
        T += each(f'T: {{a}} : {self.end} : {self.end} 1')
        O += each(f'O: {{a}} : 0 : {none} 1') + each(f'O: {{a}} : {self.end} : {none} 1')
        self._cache[fast] = ('\n'.join(out + T + O + R) + '\n', acc, each)
        return self._cache[fast]

    def write(self, path, lam, fast=None):
        fast = FAST if fast is None else fast
        text, acc, each = self._static(fast)
        R = []
        if lam:
            for s in acc:
                R += each(f'R: {{a}} : {s} : * : * {lam / GAMMA_S ** (self.H + 1):.17g}')
        with open(path, 'w') as fh:
            fh.write(text)
            if R:
                fh.write('\n'.join(R) + '\n')

    def policy(self, alphas, actions):
        bm = self.bm
        G = []
        for t in range(self.H):
            Gt = {}
            for r, ix in enumerate(bm.nodes_of):
                cols = np.array([self.sid.get((t, int(j)), -1) for j in ix])
                if (cols < 0).all():
                    continue
                M = np.where(cols[None, :] >= 0, alphas[:, np.maximum(cols, 0)], -np.inf)
                Gt[r] = (M, np.array([int(self.Aeff(int(a))[ix[0]]) for a in actions]))
            G.append(Gt)
        return G


def read_policy(path, N, dtype=np.float32):
    al, acts, k, sparse = None, [], 0, False
    with open(path) as fh:
        for line in fh:
            if line.startswith('<Vector'):
                head, _, rest = line.partition('>')
                acts.append(int(re.search(r'action="(\d+)"', head).group(1)))
                v = np.array(rest.partition('</Vector>')[0].split(), dtype=dtype)
                if al is None or k == len(al):
                    grown = np.empty((max(16, 2 * k), N), dtype)
                    if al is not None:
                        grown[:k] = al
                    al = grown
                al[k] = v
                k += 1
            elif line.startswith('<AlphaVector'):
                m = re.search(r'numVectors="(\d+)"', line)
                if m and al is None:
                    al = np.empty((int(m.group(1)), N), dtype)
            elif '<SparseVector' in line:
                sparse = True
    if not sparse:
        return (al[:k] if al is not None else np.zeros((0, N), dtype)), np.array(acts)
    al, acts = [], []
    txt = Path(path).read_text()
    for m in re.finditer(r'<SparseVector action="(\d+)" obsValue="\d+">(.*?)</SparseVector>', txt, re.S):
        v = np.zeros(N, dtype)
        for e in re.finditer(r'<Entry>\s*(\d+)\s+([-+0-9.eE]+)\s*</Entry>', m.group(2)):
            v[int(e.group(1))] = float(e.group(2))
        acts.append(int(m.group(1)))
        al.append(v)
    return np.array(al), np.array(acts)


class SarsopPolicy:
    def __init__(self, ex, alphas, actions):
        self.ex, self.al, self.acts, self.blocks = ex, alphas, np.asarray(actions), {}
        self.first_legal = ex.legal_node.argmax(1)

    def act(self, t, r, b):
        blk = self.blocks.get((t, r))
        if blk is None:
            ix = self.ex.bm.nodes_of[r]
            cols = np.array([self.ex.sid.get((t, int(j)), -1) for j in ix])
            ok = cols >= 0
            j0 = ix[0]
            acts = np.where(self.ex.legal_node[j0, self.acts], self.acts, self.first_legal[j0])
            blk = self.blocks[(t, r)] = (self.al[:, cols[ok]].astype(np.float64), ok, acts)
        M, ok, acts = blk
        return int(acts[int(np.argmax(M @ b[ok]))])


def evaluate(bm, pol):
    bm.greedy = lambda Gamma, t, r, b: pol.act(t, r, b)
    e = bm.evaluate(None, 0.0)
    return e['J_phi'], e['J_r'], e['tree']


def solve(ex, lam, timeout, precision, workdir, logfile=None):
    f = Path(workdir) / f'm_{os.getpid()}_{lam:.10g}.pomdp'
    pf = f.with_suffix('.policy')
    ex.write(f, lam)
    t0 = time.time()
    cmd = [SARSOP, str(f), '--timeout', str(timeout), '--precision', str(precision), '-o', str(pf)] + (['--fast'] if FAST else [])
    if shutil.which('stdbuf'):
        cmd = ['stdbuf', '-oL'] + cmd
    logfile = Path(logfile) if logfile else f.with_suffix('.log')
    with open(logfile, 'w') as fh:
        code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    out = logfile.read_text(errors='replace')
    secs = time.time() - t0
    rows = re.findall(r'^\s*([\d.]+)\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)', out, re.M)
    lb, ub = (float(rows[-1][3]), float(rows[-1][4])) if rows else (np.nan, np.nan)
    if not pf.exists():
        raise RuntimeError(f'SARSOP failed (exit code {code}), see {logfile}:\n' + out[-2000:])
    al, acts = read_policy(pf, ex.N)
    for x in (f, pf):
        x.unlink()
    return al, acts, lb, ub, secs


def _same(a, b):
    return abs(a - b) <= 1e-9 * (1 + abs(a))


class Cache:
    def __init__(self, folder, key, name):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.key, self.mine = key, self.folder / f'{key}__{name}.jsonl'

    def load(self):
        cols = []
        for f in sorted(self.folder.glob(f'{self.key}__*.jsonl')):
            for line in f.read_text().splitlines():
                try:
                    cols.append(json.loads(line))
                except ValueError:
                    pass
        return cols

    def add(self, col):
        with open(self.mine, 'a') as fh:
            fh.write(json.dumps(col, default=float) + '\n')
            fh.flush()
            os.fsync(fh.fileno())


def run(task, model, c, timeout=120, precision=1e-3, max_iter=20, tol=1e-4, ub_timeout=0, patience=2,
        cache=None, workdir=None, logdir=None, name='', only=None):
    t0 = time.time()
    bm = CB.BeliefModel(task, model)
    ex = Export(task, model, bm)
    R = lbs.reward_span(task)
    room = task['env']['info'].get('ceiling', 1.0) - c
    lam_hi = 2.0 * R / room if room > 0 else 1e6
    cols, log = [], []
    seen = set()

    def merge():
        for x in (cache.load() if cache else []):
            if x.get('uid') not in seen:
                seen.add(x.get('uid'))
                cols.append(x)

    tmp = None if workdir else tempfile.TemporaryDirectory()
    wd = Path(workdir) if workdir else Path(tmp.name)
    wd.mkdir(parents=True, exist_ok=True)
    try:
        def column(lam, tag, t_out=timeout, prec=precision):
            merge()
            hit = next((x for x in cols if _same(x['lam'], lam) and (tag != 'bound' or x.get('tag') == 'bound')), None)
            if hit is not None:
                print(f'  [{tag}] lam {lam:.6g}: from the cache ({hit.get("tag")}, J_phi {hit["J_phi"]:.6f} '
                      f'J_r {hit["J_r"]:.6f})', flush=True)
                return hit
            logf = Path(logdir) / f'{name}_{tag}.log' if logdir else None
            if logf:
                logf.parent.mkdir(parents=True, exist_ok=True)
                print(f'  [{tag}] lam {lam:.6g}: SARSOP running, progress in {logf}', flush=True)
            al, acts, lb, ub, secs = solve(ex, lam, t_out, prec, wd, logf)
            t1 = time.time()
            phi, r, size = evaluate(bm, SarsopPolicy(ex, al, acts))
            col = dict(lam=lam, J_phi=phi, J_r=r, L=r + lam * phi, sarsop_lb=lb, sarsop_ub=ub, sarsop_secs=round(secs, 1),
                       eval_secs=round(time.time() - t1, 1), alphas=len(al), tree=size, tag=tag, timeout=t_out,
                       uid=f'{os.getpid()}-{time.time():.6f}')
            del al
            seen.add(col['uid'])
            cols.append(col)
            if cache:
                cache.add(col)
            print(f'  [{tag}] lam {lam:.6g}: J_phi {phi:.6f} J_r {r:.6f} | SARSOP bounds on L [{lb:.6g}, {ub:.6g}] '
                  f'in {secs:.0f}s, {len(acts)} alpha-vectors, exact L {r + lam * phi:.6g} '
                  f'(evaluated in {col["eval_secs"]:.0f}s)', flush=True)
            return col

        if only == 'reward':
            column(0.0, 'reward')
            return None, None
        if only == 'satisfaction':
            column(lam_hi, 'satisfaction')
            return None, None
        column(0.0, 'reward')
        merge()
        if not any(x['J_phi'] >= c - 1e-9 for x in cols):
            column(lam_hi, 'satisfaction')
        merge()
        if not any(x['J_phi'] >= c - 1e-9 for x in cols):
            print(f'  no column reaches J_phi >= {c:g} yet: one retry at a larger multiplier with twice the time',
                  flush=True)
            column(3.0 * lam_hi, 'satisfaction_retry', 2 * timeout)
        status, it, stall = 'max_iter', 0, 0
        for it in range(max_iter):
            merge()
            best, lam, mu = CB.master(cols, c)
            if best is None:
                status = 'infeasible'
                break
            if any(_same(x['lam'], lam) for x in cols):
                status = 'converged (multiplier repeated)'
                break
            before = best[0]
            col = column(lam, f'column{it}')
            rc = col['L'] - mu
            after = CB.master(cols, c)[0][0]
            log.append(dict(iter=it, lam=lam, mu=mu, reduced_cost=rc, value=after))
            if rc <= tol * (1 + abs(mu)):
                status = 'converged'
                break
            stall = stall + 1 if after - before <= tol * (1 + abs(before)) else 0
            if stall >= patience:
                status = 'converged (master stalled)'
                break
        merge()
        best, lam, mu = CB.master(cols, c)
        if ub_timeout and best is not None:
            column(lam, 'bound', ub_timeout, precision / 10)
    finally:
        if tmp is not None:
            tmp.cleanup()
    merge()
    best, lam, mu = CB.master(cols, c)
    ub = min((x['sarsop_ub'] - x['lam'] * c for x in cols if np.isfinite(x['sarsop_ub'])), default=np.nan)
    rec = dict(method='cgcp_sarsop', c=c, status=status, iterations=it + 1, columns=len(cols), lam=lam,
               upper_bound=float(ub), secs=round(time.time() - t0, 1), timeout=timeout, precision=precision,
               states=ex.N)
    if best is not None:
        v, i, j, th = best
        rec.update(J_r=float(v), J_phi=float(th * cols[j]['J_phi'] + (1 - th) * cols[i]['J_phi']), theta=float(th),
                   mixed=bool(i != j), pair=(cols[i]['J_phi'], cols[i]['J_r'], cols[j]['J_phi'], cols[j]['J_r']))
    return rec, dict(columns=cols, iterations=log)


def main(argv=None):
    from tasks import make_task
    from experiments import model_for
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', required=True)
    ap.add_argument('--c', type=float, nargs='+', required=True)
    ap.add_argument('--root', default='runs_paper')
    ap.add_argument('--timeout', type=float, default=120, help='seconds per SARSOP call')
    ap.add_argument('--precision', type=float, default=1e-3)
    ap.add_argument('--ub-timeout', type=float, default=600, help='seconds for one last SARSOP call at the final '
                    'multiplier, to tighten the upper bound (0: skip)')
    ap.add_argument('--max-iter', type=int, default=20)
    ap.add_argument('--only', choices=('reward', 'satisfaction'), default=None,
                    help='only compute and cache this column (for the largest --c), then stop: run it first, so the '
                         'parallel runs at each threshold find it in the cache')
    ap.add_argument('--no-cache', action='store_true', help='do not read or write <root>/p2_cgcp/cache')
    ap.add_argument('--workdir', default=None, help='where the .pomdp and policy files go while SARSOP runs '
                    '(default <root>/p2_cgcp/work; five rocks needs a few GB free there)')
    a = ap.parse_args(argv)
    opts = dict(bad_rock_penalty=0.0) if a.task.startswith('rocksample') else {}
    task = make_task(a.task, variant='logic', seed=0, memory_size=2, **opts)
    model = model_for(task)
    base = Path(a.root) / 'p2_cgcp'
    workdir = a.workdir or str(base / 'work')
    logdir = base / 'logs'
    if a.only:
        c = max(a.c)
        print(f'{a.task}: the {a.only} column (c = {c:g})', flush=True)
        run(task, model, c, a.timeout, a.precision, cache=Cache(base / 'cache', a.task, f'{a.only}_{os.getpid()}'),
            workdir=workdir, logdir=logdir, name=a.task, only=a.only)
        return
    for c in a.c:
        out = base / f'{a.task}_cgcp_c{c:g}_t{a.timeout:g}'
        if (out / 'result.json').exists():
            print('done already:', out)
            continue
        print(f'{a.task}, c = {c:g}', flush=True)
        cache = None if a.no_cache else Cache(base / 'cache', a.task, f'c{c:g}_{os.getpid()}')
        rec, log = run(task, model, c, a.timeout, a.precision, max_iter=a.max_iter, ub_timeout=a.ub_timeout,
                       cache=cache, workdir=workdir, logdir=logdir, name=f'{a.task}_c{c:g}')
        rec.update(task=a.task, kind='cgcp_sarsop', exp='p2_cgcp')
        out.mkdir(parents=True, exist_ok=True)
        if rec['status'] == 'infeasible':
            (out / 'infeasible.json').write_text(json.dumps(rec, indent=1, default=float), encoding='utf-8')
            print(f'  -> no SARSOP policy reached J_phi >= {c:g} within the time limit. Nothing is marked done: rerun '
                  f'with a larger TIMEOUT (the columns found so far stay in the cache).', flush=True)
            continue
        (out / 'log.json').write_text(json.dumps(log, default=float), encoding='utf-8')
        (out / 'result.json').write_text(json.dumps(dict(job=dict(exp='p2_cgcp', kind='cgcp_sarsop', task=a.task, c=c,
                                                                   timeout=a.timeout), records=[rec]), indent=1,
                                                    default=float), encoding='utf-8')
        print(f'  -> J_r {rec.get("J_r")}, J_phi {rec.get("J_phi")}, upper bound {rec["upper_bound"]:.6g}, '
              f'{rec["status"]}, {rec["secs"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
