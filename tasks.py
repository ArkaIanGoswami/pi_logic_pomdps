"""make_task and every preset (python tasks.py --list)."""

import argparse
import json
from pathlib import Path
import rocksample
import tmaze
import twostep
import chain
import tmaze_gap
from logic import trivial_dfa

ROCK_CONFIGS = {
    3: dict(width=5, height=5, rocks=((1, 1), (3, 3), (4, 4)), start=(1, 1), horizon=32),
    5: dict(width=5, height=5, rocks=((2, 3), (4, 4), (4, 2), (3, 5), (5, 3)), start=(1, 1), horizon=40),
    7: dict(width=7, height=7, rocks=((3, 1), (1, 2), (4, 2), (7, 4), (3, 5), (4, 5), (6, 6)),
            start=(1, 4), horizon=60),
}
DELTA_LEVELS = {'tight': 'tight', 'strict': None, 'mid': 0.5, 'loose': 0.9}
CHAIN_DECISIONS = tuple(range(2, 13))
GAP_SURVEYS = (2, 3, 4, 6)
TMAZE_GRID = dict(corridor_length=(3, 5, 10, 20), slack=tuple(range(7)), start_margin=(0, 1))


def presets():
    out = {}
    for L in TMAZE_GRID['corridor_length']:
        for s in TMAZE_GRID['slack']:
            for m in TMAZE_GRID['start_margin']:
                out[f'tmaze_L{L}_s{s}_m{m}'] = dict(
                    family='tmaze', env=dict(corridor_length=L, slack=s, start_margin=m))
    for k in ROCK_CONFIGS:
        for level in DELTA_LEVELS:
            out[f'rocksample{k}_{level}'] = dict(family='rocksample', rocks=k, delta_level=level)
    out['twostep'] = dict(family='twostep', delta=0.75)
    for T in CHAIN_DECISIONS:
        out[f'chain{T}'] = dict(family='chain', env=dict(decisions=T), delta=0.25)
    for K in GAP_SURVEYS:
        for b in range(K):
            out[f'tgap_K{K}_s{b}'] = dict(family='tmaze_gap', env=dict(surveys=K, budget=b), delta=0.25)
    return out


def level_delta(env, level):
    if not isinstance(level, str):
        return float(level)
    value = DELTA_LEVELS[level]
    if value == 'tight':
        return 1.0 - (env['info']['ceiling'] - 0.001)
    if value is None:
        return rocksample.strictest_delta(env['config'])
    return float(value)


def _resolve(spec):
    if isinstance(spec, dict):
        return dict(spec)
    spec = str(spec)
    if spec.endswith('.json'):
        return json.loads(Path(spec).read_text(encoding='utf-8'))
    variant = None
    if spec.endswith('_free'):
        spec, variant = spec[:-5], 'free'
    table = presets()
    if spec in table:
        cfg = table[spec]
    elif spec == 'tmaze':
        cfg = dict(family='tmaze', env={})
    elif spec.startswith('rocksample') and spec[10:].isdigit():
        cfg = dict(family='rocksample', rocks=int(spec[10:]), delta_level='strict')
    else:
        raise KeyError(f'unknown task {spec!r}; see `python tasks.py --list`')
    if variant:
        cfg['variant'] = variant
    cfg['name'] = spec + ('_free' if variant else '')
    return cfg


def make_task(spec, variant=None, seed=None, memory_size=None, objective=None, delta=None, **env_overrides):
    cfg = _resolve(spec)
    override = objective or cfg.get('objective')
    variant = variant or cfg.get('variant', 'logic')
    seed = int(cfg.get('seed', 0) if seed is None else seed)
    zsize = int(memory_size or cfg.get('memory_size', 2))
    family = cfg['family']
    if family == 'tmaze':
        env_cfg = {**cfg.get('env', {}), **env_overrides}
        fixed_layout = 'layout_seed' in env_cfg
        env_cfg.setdefault('layout_seed', seed)
        env, dfa = tmaze.make_env(env_cfg), tmaze.make_spec()
        objective, delta = ('P1' if variant == 'logic' else 'P0'), None
    elif family == 'rocksample':
        k = int(cfg.get('rocks', 5))
        env_cfg = {**ROCK_CONFIGS[k], **cfg.get('env', {}), **env_overrides}
        env, dfa = rocksample.make_env(env_cfg), rocksample.make_spec()
        delta = cfg.get('delta') if delta is None else delta
        if delta is None:
            delta = level_delta(env, cfg.get('delta_level', 'strict'))
        objective = 'P2' if variant == 'logic' else 'P0'
        if 1 - delta > env['info']['ceiling'] + 1e-12:
            print(f'warning: 1 - delta = {1 - delta:.3f} exceeds the ceiling {env["info"]["ceiling"]:.3f}')
    elif family == 'twostep':
        env, dfa = twostep.make_env({**cfg.get('env', {}), **env_overrides}), twostep.make_spec()
        delta = float(cfg.get('delta', 0.75) if delta is None else delta)
        objective = 'P2' if variant == 'logic' else 'P0'
    elif family == 'chain':
        env, dfa = chain.make_env({**cfg.get('env', {}), **env_overrides}), chain.make_spec()
        delta = float(cfg.get('delta', 0.25) if delta is None else delta)
        objective = 'P2' if variant == 'logic' else 'P0'
    elif family == 'tmaze_gap':
        env, dfa = tmaze_gap.make_env({**cfg.get('env', {}), **env_overrides}), tmaze_gap.make_spec()
        delta = float(cfg.get('delta', 0.25) if delta is None else delta)
        objective = 'P2' if variant == 'logic' else 'P0'
    else:
        raise KeyError(family)
    if variant not in ('logic', 'free'):
        raise ValueError("variant must be 'logic' or 'free'")
    if variant == 'free':
        dfa = trivial_dfa()
    if override:
        objective = override
    z0 = cfg.get('initial_memory', [1.0] + [0.0] * (zsize - 1))
    if len(z0) != zsize or abs(sum(z0) - 1) > 1e-9:
        raise ValueError('initial_memory must be a probability vector of length memory_size')
    saved = {k: v for k, v in cfg.items() if k != 'env'}
    saved.update(family=family, variant=variant, seed=seed, memory_size=zsize,
                 env={k: v for k, v in env['config'].items()})
    if family == 'tmaze' and not fixed_layout:
        saved['env'].pop('layout_seed')
    if family in ('rocksample', 'twostep', 'chain', 'tmaze_gap'):
        saved['delta'] = float(delta)
    if override:
        saved['objective'] = override
    return dict(name=cfg.get('name', family), family=family, variant=variant, objective=objective,
                env=env, dfa=dfa, horizon=env['horizon'], gamma=env['gamma'],
                delta=None if delta is None else float(delta), memory_size=zsize,
                initial_memory=tuple(float(x) for x in z0), seed=seed, config=saved)


def save_config(task, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(task['config'], indent=1, default=list), encoding='utf-8')


def describe(task):
    e = task['env']
    line = (f"{task['name']}: {task['family']} {task['variant']} {task['objective']} "
            f"H={task['horizon']} gamma={task['gamma']} |Z|={task['memory_size']} "
            f"|A|={len(e['actions'])} automaton={task['dfa']['formula']}")
    if task['family'] == 'tmaze':
        lay = e['layout']
        line += f" slack={e['info']['slack']} keys={lay['keys']}"
    else:
        line += f" delta={task['delta']} ceiling={e['info']['ceiling']:.4f}"
    return line


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--show')
    ap.add_argument('--export')
    a = ap.parse_args()
    if a.list:
        print('\n'.join(presets()))
    if a.show:
        print(describe(make_task(a.show)))
    if a.export:
        for name in presets():
            for variant in ('logic', 'free'):
                t = make_task(name, variant=variant)
                save_config(t, Path(a.export) / f"{name}{'_free' if variant == 'free' else ''}.json")
        print(f'wrote {2 * len(presets())} configs to {a.export}')
