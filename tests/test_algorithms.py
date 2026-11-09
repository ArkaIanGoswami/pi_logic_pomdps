"""Tests of the algorithms (python -m pytest -q tests)."""

import json
import numpy as np
import pytest
import cpi
import experiments
import learn
from product import compile_product
from tasks import level_delta, make_task


def test_stage_identity_and_monotone_p1():
    task = make_task('tmaze_L5_s2_m1', seed=1, memory_size=2)
    model = compile_product(task)
    out = cpi.solve(task, model, dict(alpha=0.9, start='copy', ties='random'), rng=np.random.default_rng(1))['P1']
    assert max(l['identity_err'] for l in out['log'][1:]) < 1e-10
    assert experiments.monotone(out['log'])
    assert out['J']['phi'] == pytest.approx(1.0)


def test_joint_step_feasible_and_lp_optimal():
    task = make_task('rocksample3_tight', seed=2)
    model = compile_product(task)
    c = 1 - task['delta']
    out = cpi.solve(task, model, dict(alpha=0.9), [('tight', c)])['tight']
    p2 = [l for l in out['log'] if l['phase'] == 'P2']
    assert min(l['min_phi'] for l in p2) >= c - 1e-9
    assert max(l['gap'] for l in p2) < 1e-8
    assert sum(l['lam_pos'] for l in p2) > 0
    assert experiments.monotone(out['log'])
    assert cpi.evaluate(task, model, out['pol'])['J_phi'] >= c - 1e-9


def test_memory_blind_start_returns_the_memoryless_value():
    vals = []
    for Z in (1, 2):
        task = make_task('tmaze_L10_s0_m1', seed=3, memory_size=Z)
        vals.append(cpi.solve(task, compile_product(task), dict(alpha=0.9, start='uniform'))['P1']['J']['phi'])
    assert vals[0] == pytest.approx(vals[1], abs=1e-12)


def test_two_step_example():
    task = make_task('twostep', seed=0, memory_size=1)
    model = compile_product(task)
    c = 1 - task['delta']
    joint = cpi.solve(task, model, dict(alpha=0.9), [('c', c)])['c']['J']
    assert joint['phi'] >= c - 1e-9 and joint['r'] <= 0.25 + 1e-9
    p1 = cpi.solve(dict(task, objective='P1'), model, dict(alpha=0.9))['P1']['pol']
    lag = cpi.lagrangian(task, model, c, p1, cpi.initial_policy(task, model, 'dirichlet', 0), {})
    assert lag['status'] == 'mixed'
    assert lag['J_phi'] == pytest.approx(c, abs=1e-9) and lag['J_r'] == pytest.approx(0.75, abs=1e-6)


def test_learned_model_on_deterministic_maze_matches_exact():
    task = make_task('tmaze_L5_s0_m1', seed=0, memory_size=2)
    model = compile_product(task)
    recs, _, _ = learn.run(task, dict(m=1, alpha=0.9, start='copy', ties='random', holdout=2000), [], 0, true_model=model)
    assert recs[0]['J_phi'] == pytest.approx(1.0) and recs[0]['latent_states'] == len({s for s, _, _ in model['nodes']})
    assert 0.99 < recs[0]['holdout_lower'] <= 1.0


def test_policy_round_trip(tmp_path):
    task = make_task('rocksample3_tight', seed=0)
    model = compile_product(task)
    pol = cpi.initial_policy(task, model, 'dirichlet', 0)
    cpi.save_policy(tmp_path / 'p.npz', model, pol, task)
    back = cpi.load_policy(tmp_path / 'p.npz')
    assert back['rows'] == model['rows'] and np.allclose(back['pol'], pol, atol=1e-6)
    y, q = model['rows'][5]
    assert np.allclose(cpi.as_function(back)(3, y, q, 1), pol[3, 5, 1], atol=1e-6)


def test_runner_writes_results(tmp_path):
    job = experiments._job('lagrangian', 'lagrangian', 'twostep', 0, Z=1, alpha=0.9, inner_alpha=1.0)
    jid, status, _ = experiments.run_job((job, tmp_path))
    assert status == 'done'
    rec = json.loads((tmp_path / 'lagrangian' / jid / 'result.json').read_text())['records'][0]
    assert rec['J_r'] == pytest.approx(0.75, abs=1e-6)


def test_environment_option_reaches_the_task(tmp_path):
    job = experiments._job('penalty_ablation', 'cpi', 'rocksample3_tight', 0, variant='free', alpha=0.9,
                           start='dirichlet', ties='index')
    job = experiments.with_options(job, bad_rock_penalty=0.0)
    assert 'pen0.0' in job['id'] and job['start'] == 'dirichlet'
    jid, status, _ = experiments.run_job((job, tmp_path))
    assert status == 'done'
    rec = json.loads((tmp_path / 'penalty_ablation' / jid / 'result.json').read_text())['records'][0]
    assert rec['bad_rock_penalty'] == 0.0
    assert rec['logic_J_phi'] < 0.5


def test_naive_step_stays_feasible_and_matches_joint_when_slack():
    task = make_task('rocksample3_tight', seed=0, memory_size=2)
    model = compile_product(task)
    cfg = dict(alpha=0.9, start='dirichlet', ties='index')
    levels = [('tight', 0.874), ('loose', 0.1)]
    joint = cpi.solve(task, model, cfg, levels)
    naive = cpi.solve(task, model, dict(cfg, p2_step='naive'), levels)
    for name, c in levels:
        assert naive[name]['J']['phi'] >= c - 1e-9
        assert all(l['identity_err'] < 1e-9 for l in naive[name]['log'] if l['phase'] == 'P2')
    assert naive['loose']['J']['r'] == pytest.approx(joint['loose']['J']['r'], abs=1e-9)
