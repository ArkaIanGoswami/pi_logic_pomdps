"""Tests of the environments and the infrastructure."""

import os
import sys
from collections import deque
import numpy as np
import pytest
from policies import dirichlet_row, load_policy, random_policy, save_policy, table_policy, uniform_policy
from product import compile_product, evaluate
from simulate import episode_summary, rollout
from tasks import make_task
import rocksample
import tmaze

SMALL = ['tmaze_L3_s0_m0', 'tmaze_L5_s2_m1', 'tmaze_L5_s2_m1_free', 'rocksample3_strict',
         'rocksample3_strict_free']


@pytest.mark.parametrize('name', SMALL)
def test_kernels_are_probabilities(name):
    task = make_task(name, seed=1)
    m = compile_product(task)
    assert abs(m['initial'].sum() - 1) < 1e-12
    for P in m['P']:
        assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)
    assert m['legal'].any(axis=1).all()


def _earliest_acceptance(task):
    m = compile_product(task)
    frontier = set(np.flatnonzero(m['initial'] > 0))
    for t in range(task['horizon'] + 1):
        if any(m['accepting'][j] for j in frontier):
            return t
        nxt = set()
        for j in frontier:
            for a in np.flatnonzero(m['legal'][m['row_of_node'][j]]):
                nxt.update(m['P'][a].getrow(j).indices.tolist())
        frontier = nxt
    return None


@pytest.mark.parametrize('L,a,m,seed', [(3, 0, 0, 0), (5, 0, 1, 2), (6, 2, 1, 5), (10, 1, 0, 3)])
def test_shortest_accepting_path(L, a, m, seed):
    task = make_task('tmaze', seed=seed, corridor_length=L, branch_length=a, start_margin=m, slack=3)
    assert _earliest_acceptance(task) == L + a + 5 == task['env']['info']['shortest_path']
    with pytest.raises(ValueError):
        make_task('tmaze', corridor_length=L, branch_length=a, horizon=L + a + 4)


def _plan(task, cue, delay):
    lay, cfg = task['env']['layout'], task['env']['config']
    side = 2 if cue == 0 else 3
    acts = {0: [], 1: [tmaze.action_index('pick')], 2: [1, 0]}[delay]
    for i in range(cfg['corridor_length']):
        if i == lay['keys'][cue][1]:
            acts.append(tmaze.action_index('pick'))
        acts.append(0)
    acts += [0] + [side] * cfg['branch_length'] + [tmaze.action_index('open'), side, side]
    return acts


def _scripted(task, delays, memory):
    env, dfa = task['env'], task['dfa']
    A, Z = len(env['actions']), task['memory_size']
    table, conflict = {}, False
    for cue in (0, 1):
        (s, y, _), = [i for i in env['initial'] if i[0][0] == cue]
        q, z = dfa['next'](dfa['initial'], env['labels'](y)), 0
        for t, a in enumerate(_plan(task, cue, delays[cue])[:task['horizon']]):
            zn = cue if memory else 0
            row = np.zeros((A, Z))
            row[a, zn] = 1.0
            key = (t, y, q, z)
            if key in table and not np.array_equal(table[key], row):
                conflict = True
            table.setdefault(key, row)
            (s, y, _, _), = env['step'](s, a)
            q, z = dfa['next'](q, env['labels'](y)), zn
    return table_policy(task, table), conflict


def _J(task, pi):
    return evaluate(task, compile_product(task), pi)


@pytest.mark.parametrize('m', [0, 1])
def test_one_memory_bit_solves_at_zero_slack(m):
    task = make_task('tmaze', seed=4, corridor_length=6, start_margin=m, slack=0, memory_size=2)
    pi, _ = _scripted(task, (0, 0), memory=True)
    assert _J(task, pi)['J_phi'] == pytest.approx(1.0)
    free = make_task('tmaze', variant='free', seed=4, corridor_length=6, start_margin=m, slack=0, memory_size=2)
    assert _J(free, _scripted(free, (0, 0), memory=True)[0])['J_r'] == pytest.approx(1.0)


def test_memoryless_clock_needs_slack_two():
    for s, expected in [(0, 0.5), (1, 0.5), (2, 1.0), (4, 1.0)]:
        task = make_task('tmaze', seed=2, corridor_length=5, start_margin=1, slack=s, memory_size=1)
        pi, conflict = _scripted(task, (0, 2), memory=False)
        assert not conflict
        assert _J(task, pi)['J_phi'] == pytest.approx(expected)


def test_no_delay_possible_without_start_margin():
    task = make_task('tmaze', seed=0, start_margin=0, memory_size=1)
    y0 = task['env']['initial'][0][1]
    assert task['env']['legal'](y0) == (0,)
    noop = make_task('tmaze', seed=0, start_margin=0, slack=1, allow_noop=True, memory_size=1)
    pi, conflict = _scripted(noop, (0, 1), memory=False)
    assert not conflict and _J(noop, pi)['J_phi'] == pytest.approx(1.0)


def test_fixed_side_memoryless():
    task = make_task('tmaze', variant='free', seed=0, corridor_length=5, slack=0, memory_size=1)
    pi, _ = _scripted(task, (0, 0), memory=False)
    r = _J(task, pi)
    assert r['J_r'] == pytest.approx(0.0)


def test_cue_is_never_observed_after_reset():
    task = make_task('tmaze_L10_s3_m1', seed=7)
    m = compile_product(task)
    first = set(np.flatnonzero(m['initial'] > 0))
    for j, (s, y, q) in enumerate(m['nodes']):
        if j not in first:
            assert not y[1].startswith('cue')
    assert all(set(task['env']['labels'](y)) <= {'k', 'd', 'g_c', 'fail'} for _, y, _ in m['nodes'])


def test_rocksample_ceilings_and_deltas():
    expect = {3: (0.875, 0.2), 5: (0.96875, 0.1), 7: (1 - 2 ** -7, 0.1)}
    for k, (ceil, delta) in expect.items():
        task = make_task(f'rocksample{k}_strict')
        assert task['env']['info']['ceiling'] == pytest.approx(ceil)
        assert task['delta'] == pytest.approx(delta)
        assert make_task(f'rocksample{k}_mid')['delta'] == 0.5
        assert make_task(f'rocksample{k}_loose')['delta'] == 0.9


@pytest.mark.parametrize('name', ['tmaze_L5_s2_m1', 'rocksample3_strict'])
def test_rollouts_agree_with_exact_evaluation(name):
    task = make_task(name, seed=3)
    pi = random_policy(task, 3)
    exact = evaluate(task, compile_product(task), pi)
    rng = np.random.default_rng(0)
    eps = [episode_summary(task, rollout(task, pi, rng)) for _ in range(4000)]
    assert abs(np.mean([e['accepted'] for e in eps]) - exact['J_phi']) < 4 * np.sqrt(.25 / 4000) + 1e-3
    lo, hi = task['env']['reward_range']
    sd = (hi - lo) / 2 * np.sqrt(task['horizon'])
    assert abs(np.mean([e['discounted_return'] for e in eps]) - exact['J_r']) < 4 * sd / np.sqrt(4000)


def test_policy_files_round_trip(tmp_path):
    task = make_task('rocksample3_strict')
    A, Z = len(task['env']['actions']), task['memory_size']
    y0 = task['env']['initial'][0][1]
    row = np.eye(A * Z)[3].reshape(A, Z)
    save_policy(tmp_path / 'p.pkl', {(0, y0, 0, 0): row}, Z, seed=5)
    pi = load_policy(task, tmp_path / 'p.pkl')
    assert np.array_equal(pi(0, y0, 0, 0), row)
    assert np.array_equal(pi(1, y0, 0, 0), dirichlet_row(5, (1, y0, 0, 0), A, Z))


def test_figures_and_animation(tmp_path):
    import env_animate
    import env_visualize
    env_visualize.main(['tmaze_L5_s2_m1', 'rocksample3', '--outdir', str(tmp_path)])
    task = make_task('tmaze_L3_s0_m0')
    ep = rollout(task, uniform_policy(task), np.random.default_rng(0))
    out = env_animate.animate(task, ep, tmp_path / 'a.gif', fps=4, hold=1, dpi=60)
    assert os.path.getsize(out) > 0 and len(list(tmp_path.glob('*.png'))) == 2
