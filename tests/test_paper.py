"""Tests of the instances and methods of the paper."""

import numpy as np
import pytest
import chain
import cpi
from experiments import model_for
from tasks import make_task


@pytest.mark.parametrize('T,c', [(2, 0.25), (3, 0.5), (5, 0.75)])
def test_chain_closed_form(T, c):
    task = make_task(f'chain{T}', seed=0, memory_size=1, delta=1 - c)
    model = model_for(task)
    x = c ** (1 / T)
    pol = np.zeros((T, len(model['rows']), 1, 2, 1))
    pol[..., 0, 0], pol[..., 1, 0] = 1 - x, x
    J = cpi.evaluate(task, model, pol)
    assert J['J_phi'] == pytest.approx(c, abs=1e-12)
    assert J['J_r'] == pytest.approx(chain.v2(T, c), abs=1e-12)
    assert chain.v2(T, c) <= (np.log(1 / c) / T) ** T + 1e-15


def test_chain_two_decisions_matches_twostep():
    assert chain.v2(2, 0.25) == pytest.approx(0.25)
    assert chain.v2_bar(0.25) == pytest.approx(0.75)


import tmaze_gap as tg
from itertools import product
from scipy.optimize import minimize


def test_tgap_values():
    assert tg.v2(0.4, 2, 0) == pytest.approx(0.6)
    assert tg.v2(0.75, 4, 3) == pytest.approx(0.25)
    assert np.isnan(tg.v2(0.75, 4, 2))
    vals = [tg.v2(0.75, K, 0) for K in (2, 3, 4, 6)]
    assert all(a > b for a, b in zip(vals, vals[1:])) and vals[-1] < 1e-5
    assert tg.v2(0.75, 2, 0) == pytest.approx(0.0428932, abs=1e-6)


def test_tgap_equal_scan_probabilities_are_optimal():
    def p_le(y, s):
        return sum(np.prod([v if b else 1 - v for v, b in zip(y, bits)])
                   for bits in product((0, 1), repeat=len(y)) if sum(bits) <= s)
    rng = np.random.default_rng(0)
    for K, s, phi in ((3, 1, 0.6), (4, 2, 0.3)):
        best = 0.0
        for _ in range(15):
            r = minimize(lambda y: -np.prod(y), rng.uniform(0, 1, K), method='SLSQP', bounds=[(0, 1)] * K,
                         constraints=[dict(type='ineq', fun=lambda y: p_le(y, s) - phi)], options=dict(ftol=1e-14))
            if p_le(r.x, s) >= phi - 1e-9:
                best = max(best, float(np.prod(r.x)))
        assert best == pytest.approx(tg._branch_frontier(K, s, phi), rel=1e-5)


def test_tgap_value_matches_direct_search():
    import bruteforce
    c = 0.6
    task = make_task('tgap_K2_s0', seed=0, memory_size=2, delta=1 - c)
    best, _, _ = bruteforce.search(task, model_for(task), c, restarts=12, scale=0.5, seed=1)
    assert best == pytest.approx(tg.v2(c, 2, 0), abs=1e-5)


def test_tgap_memory_and_coin_start():
    for Z, want in ((1, 0.5), (2, 1.0)):
        task = make_task('tgap_K4_s0', seed=0, memory_size=Z, objective='P1')
        model = model_for(task)
        r = cpi.solve(task, model, dict(alpha=0.9, start='copycoin', ties='random'), rng=np.random.default_rng(0))['P1']
        assert r['J']['phi'] == pytest.approx(want, abs=1e-6)


import lbs


def test_bisection_reaches_v2_bar_on_chain_and_beats_bracket_bound():
    task = make_task('chain4', seed=0, memory_size=1, delta=0.25)
    model = model_for(task)
    out = lbs.bisection(task, model, 0.75, cpi.initial_policy(task, model, 'dirichlet', 0),
                            dict(alpha=0.9, inner_alpha=1.0, ties='index'), steps=14)
    assert out['status'] == 'mixed'
    assert out['J_phi'] == pytest.approx(0.75, abs=1e-9) and out['J_r'] == pytest.approx(0.25, abs=1e-9)
    assert out['J_r'] >= out['bracket_J_r'] - 1e-12
    lo, hi = out['J_lo'], out['J_hi']
    bound = (0.25 - out['theta'] * out['eps_hi'] - (1 - out['theta']) * out['eps_lo'] - out['certificate'])
    assert out['bracket_J_r'] >= bound - 1e-9
    vals = [v for _, v in out['curve'] if v == v]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_incumbent_keeps_only_hull_policies():
    inc = lbs.Incumbent(0.5)
    pol = np.zeros((1, 1, 1, 1, 1))
    for phi, r in ((0.0, 1.0), (1.0, 0.0), (0.4, 0.2), (0.6, 0.3), (0.5, 0.6)):
        inc.add(dict(phi=phi, r=r), pol, 'x')
    assert set(inc.pols) == {0, 1, 4}
    assert inc.best[0] == pytest.approx(0.6)


def test_exponentiated_gradient_runs_and_reports_average():
    task = make_task('tgap_K2_s0', seed=0, memory_size=2, delta=0.25)
    model = model_for(task)
    out = lbs.exponentiated_gradient(task, model, 0.75, cpi.initial_policy(task, model, 'copycoin', 0),
                                         dict(alpha=0.9, inner_alpha=0.9, ties='random', seed=0), iters=8)
    assert out['status'] == 'done' and len(out['lams']) == 8
    assert out['avg_J_phi'] == pytest.approx(np.mean(out['iter_J_phi']))
    assert out['J_phi'] >= 0.75 - 1e-9


def test_p2_from_converged_p1():
    task = make_task('chain3', seed=0, memory_size=2, delta=0.25)
    model = model_for(task)
    r = cpi.solve(task, model, dict(alpha=1.0, lazy=True, p2_from='converged'), [('c', 0.75)])['c']
    assert r['J']['phi'] >= 0.75 - 1e-9 and r['J']['r'] == pytest.approx(0.0, abs=1e-12)


def test_joint_rule_never_buys_back_roundoff():
    from experiments import max_drop
    c, seed = 0.9, 15
    task = make_task('tgap_K4_s0', seed=seed, memory_size=4, delta=1 - c)
    model = model_for(task)
    res = cpi.solve(task, model, dict(alpha=0.9, start='copycoin', ties='random', p2_step='joint'), [('c', c)],
                    rng=np.random.default_rng(seed))['c']
    assert max_drop(res['log']) < 1e-8
    assert min(l['min_phi'] for l in res['log'] if l['phase'] == 'P2') >= c - cpi.FEAS


def test_oracle_uses_hull_policies():
    import lbs
    c = 0.75
    task = make_task('tgap_K4_s2', seed=0, memory_size=2, delta=1 - c)
    model = model_for(task)
    start = cpi.initial_policy(task, model, 'copycoin', seed=0)
    r = lbs.bisection(task, model, c, start, dict(alpha=0.9, inner_alpha=0.9, start='copycoin', ties='random', seed=0))
    assert r['lam_hi'] == pytest.approx(1.0, abs=1e-3)
    assert max(r['eps_lo'], r['eps_hi']) < 1e-9
    assert r['J_r'] == pytest.approx(0.25, abs=1e-9)


def test_memoryless_route_policy():
    import tmaze_route
    def value(L, slack, margin, seed=0):
        task = make_task(f'tmaze_L{L}_s0_m{margin}', seed=seed, memory_size=1, slack=slack)
        return tmaze_route.route_value(task, model_for(task))
    for L in (3, 5):
        for seed in (0, 1, 2):
            v, status = value(L, 2, 1, seed)
            assert status == 'ok' and v == pytest.approx(1.0, abs=1e-12)
    assert value(3, 1, 1)[0] is None
    assert value(3, 4, 0)[0] is None
