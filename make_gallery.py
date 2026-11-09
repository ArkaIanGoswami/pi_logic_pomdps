"""The animations of learned policies shown in the README - Entirely Claude-generated (but verified)"""

import argparse
from pathlib import Path
import numpy as np
import cpi
from env_animate import animate
from experiments import _job, env_overrides, run_job
from rocksample import first_check
from simulate import rollout
from tasks import make_task

ROOT = 'runs_gallery'
OUT = 'examples'
ROCK = dict(start='dirichlet', ties='index', alpha=0.9, bad_rock_penalty=0.0, save=True)
GALLERY = [
    ('learned_tmaze_L5_s0_m1_Z2', _job('gallery', 'cpi', 'tmaze_L5_s0_m1', 0, Z=2, alpha=0.9, start='copy',
                                       ties='random', save=True), 'cues'),
    ('learned_tmaze_L3_s4_m1_Z1', _job('gallery', 'cpi', 'tmaze_L3_s4_m1', 0, Z=1, alpha=0.9, start='copy',
                                       ties='random', save=True), 'cues'),
    ('learned_rocksample3_c0.86_cpi', _job('gallery', 'cpi', 'rocksample3_tight', 0, Z=2, levels=[0.86], **ROCK),
     'check'),
    ('learned_rocksample3_c0.86_lbscpi', _job('gallery', 'bisect', 'rocksample3_tight', 0, Z=2, c=0.86,
                                              inner_alpha=0.9, **ROCK), 'check'),
]
FIVE_ROCKS = [
    ('learned_rocksample5_c0.96_lbscpi', _job('gallery', 'bisect', 'rocksample5_tight', 0, Z=2, c=0.96,
                                              inner_alpha=0.9, **ROCK), 'check'),
]


def saved_policy(folder):
    files = sorted(Path(folder).glob('policy_*.npz'))
    if not files:
        raise FileNotFoundError(f'no saved policy in {folder}')
    for name in ('policy_bisect.npz', 'policy_mixture.npz'):
        if (Path(folder) / name).exists():
            return Path(folder) / name
    return files[-1]


def pick_episodes(task, pi, how, tries=200):
    chosen = {}
    for e in range(tries):
        ep = rollout(task, pi, np.random.default_rng(e))
        if how == 'cues':
            chosen.setdefault(f"cue{ep[0]['s'][0]}", (e, ep))
            if len(chosen) == 2:
                break
        else:
            checks = any(x['a'] >= first_check() for x in ep)
            if checks and ep[-1]['q_next'] in task['dfa']['accepting']:
                return {f'ep{e}': (e, ep)}
    if not chosen:
        chosen = {'ep0': (0, rollout(task, pi, np.random.default_rng(0)))}
    return chosen


def make(name, job, how, root=ROOT, out=OUT):
    jid, status, secs = run_job((job, root))
    folder = Path(root) / job['exp'] / job['id']
    if status == 'error':
        raise RuntimeError((folder / 'error.txt').read_text())
    task = make_task(job['task'], variant=job['variant'], seed=job['seed'], memory_size=job['Z'], **env_overrides(job))
    saved = cpi.load_policy(saved_policy(folder))
    task = dict(task, memory_size=saved['pol'].shape[2], initial_memory=tuple(saved['initial_memory']))
    pi = cpi.as_function(saved)
    for tag, (e, ep) in pick_episodes(task, pi, how).items():
        path = Path(out) / f'{name}_{tag}.gif'
        animate(task, ep, path, fps=3, annotate=True)
        print(f'{path}  (job {status} in {secs:.0f} s, episode seed {e}, {len(ep)} steps)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--five-rocks', action='store_true', help='also LBS + CPI on five rocks (about 10 minutes)')
    ap.add_argument('--only', default='', help='comma list of name prefixes, e.g. tmaze,rocksample3')
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args(argv)
    entries = GALLERY + (FIVE_ROCKS if a.five_rocks else [])
    if a.only:
        keep = tuple('learned_' + p for p in a.only.split(','))
        entries = [x for x in entries if x[0].startswith(keep)]
    for name, job, how in entries:
        make(name, job, how, a.root, a.out)


if __name__ == '__main__':
    main()
