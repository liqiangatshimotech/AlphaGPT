"""Walk-forward genetic search over legal postfix alpha formulas.

This is deliberately separate from the policy-gradient trainer.  It evaluates
small candidate pools, ranks by multi-window, top-one execution metrics, and
only exposes the final frozen candidate to the test window.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from .data_loader import CryptoDataLoader
from .research_train import causal_features, metrics
from .vocab import FORMULA_VOCAB as V, FORMULA_VOCAB_VERSION, load_formula
from .ops import OPS_CONFIG
from .vm import StackVM


ARITY = [0] * V.feature_count + [x[2] for x in OPS_CONFIG]
OPS_BY_ARITY = {a: [V.feature_count + i for i, x in enumerate(OPS_CONFIG) if x[2] == a and x[0] != 'JUMP'] for a in (1, 2, 3)}


def random_formula(rng, max_len=12):
    """Sample a legal RPN formula, rejecting only at the grammar boundary."""
    for _ in range(1000):
        depth, seq = 0, []
        for pos in range(max_len):
            remaining = max_len - pos - 1
            choices = [t for t, a in enumerate(ARITY)
                       if depth >= a and 1 <= depth + 1 - a <= 1 + 2 * remaining]
            if not choices:
                break
            t = rng.choice(choices)
            seq.append(t)
            depth += 1 - ARITY[t]
            if depth == 1 and rng.random() < 0.25:
                break
        if depth == 1:
            try:
                return tuple(load_formula(seq))
            except ValueError:
                pass
    raise RuntimeError('could not sample a legal formula')


def to_tree(seq):
    stack = []
    for token in seq:
        a = ARITY[token]
        children = tuple(stack.pop() for _ in range(a))[::-1]
        stack.append((token, children))
    if len(stack) != 1:
        raise ValueError('invalid postfix formula')
    return stack[0]


def from_tree(node):
    token, children = node
    out = []
    for child in children:
        out.extend(from_tree(child))
    out.append(token)
    return tuple(out)


def paths(node, prefix=()):
    yield prefix, node
    for i, child in enumerate(node[1]):
        yield from paths(child, prefix + (i,))


def replace_path(node, path, replacement):
    if not path:
        return replacement
    token, children = node
    i = path[0]
    new_children = list(children)
    new_children[i] = replace_path(new_children[i], path[1:], replacement)
    return token, tuple(new_children)


def mutate(seq, rng, max_len=12):
    tree = to_tree(seq)
    all_paths = list(paths(tree))
    path, node = rng.choice(all_paths)
    token, children = node
    if not children:
        replacement = (rng.randrange(V.feature_count), ())
    else:
        choices = OPS_BY_ARITY[len(children)]
        replacement = (rng.choice(choices), children)
    candidate = from_tree(replace_path(tree, path, replacement))
    return tuple(load_formula(list(candidate))) if len(candidate) <= max_len else seq


def crossover(left, right, rng, max_len=12):
    a, b = to_tree(left), to_tree(right)
    pa, na = rng.choice(list(paths(a)))
    pb, nb = rng.choice(list(paths(b)))
    ca = from_tree(replace_path(a, pa, nb))
    cb = from_tree(replace_path(b, pb, na))
    candidates = [ca, cb]
    rng.shuffle(candidates)
    for candidate in candidates:
        if len(candidate) <= max_len:
            try:
                return tuple(load_formula(list(candidate)))
            except ValueError:
                pass
    return left


def percentile_threshold(factor, start, end, q=0.80):
    values = factor[:, start:end].flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values, q))


def score_formula(seq, vm, feat, raw, target, windows, cache):
    if seq in cache:
        return cache[seq]
    factor = vm.execute(seq, feat)
    if factor is None or not torch.isfinite(factor).all():
        cache[seq] = (-10.0, {})
        return cache[seq]
    threshold = percentile_threshold(factor, windows[0][0], windows[0][1])
    reports = [metrics(factor, raw, target, a, b, threshold=threshold, top_k=1)
               for a, b in windows]
    reward = float(torch.tensor([x['reward'] for x in reports]).median())
    # Prefer simple formulas when their robust score is comparable.
    reward -= 0.001 * len(seq)
    cache[seq] = (reward, {'threshold': threshold, 'windows': reports})
    return cache[seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--population', type=int, default=128)
    ap.add_argument('--generations', type=int, default=12)
    ap.add_argument('--elite', type=int, default=16)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rng = random.Random(42)
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=False, exist_ok=False)
    loader = CryptoDataLoader(); loader.load_data()
    raw, target = loader.raw_data_cache, loader.target_ret
    feat = causal_features(raw); vm = StackVM()
    T = target.shape[1]; cut, val = int(T * .6), int(T * .8)
    train_windows = [(20, int(cut * .55)), (int(cut * .25), int(cut * .78)), (int(cut * .55), cut)]
    val_windows = [(cut, cut + (val-cut)//3), (cut + (val-cut)//3, cut + 2*(val-cut)//3), (cut + 2*(val-cut)//3, val)]
    population = {random_formula(rng) for _ in range(args.population)}
    cache = {}
    best_train = []
    for generation in range(args.generations):
        ranked = sorted(((score_formula(s, vm, feat, raw, target, train_windows, cache)[0], s) for s in population), reverse=True)
        elites = [s for _, s in ranked[:args.elite]]
        best_train = ranked[:args.elite]
        row = {'generation': generation + 1, 'best_train_reward': ranked[0][0], 'median_elite_reward': float(torch.tensor([x[0] for x in ranked[:args.elite]]).median()), 'population': len(population), 'evaluated': len(cache)}
        with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        next_population = set(elites)
        while len(next_population) < args.population:
            if rng.random() < .30:
                child = crossover(rng.choice(elites), rng.choice(elites), rng)
            else:
                child = rng.choice(elites)
            if rng.random() < .85:
                child = mutate(child, rng)
            next_population.add(child)
        population = next_population
    # Validation selection is frozen after the training search.
    val_ranked = []
    for _, seq in best_train:
        factor = vm.execute(seq, feat)
        threshold = percentile_threshold(factor, train_windows[0][0], train_windows[0][1])
        reports = [metrics(factor, raw, target, a, b, threshold=threshold, top_k=1) for a, b in val_windows]
        score = float(torch.tensor([x['reward'] for x in reports]).median())
        val_ranked.append((score, seq, threshold, reports))
    val_ranked.sort(reverse=True)
    _, formula, threshold, validation = val_ranked[0]
    factor = vm.execute(formula, feat)
    report = {'config': vars(args), 'seed': 42, 'split_indices': [20, cut, val, T], 'train_windows': train_windows, 'validation_windows': val_windows, 'formula': list(formula), 'threshold': threshold, 'vocab_version': FORMULA_VOCAB_VERSION, 'token_names': list(V.token_names), 'train': score_formula(formula, vm, feat, raw, target, train_windows, cache)[1]['windows'], 'validation': validation, 'test': metrics(factor, raw, target, val, T, threshold=threshold, top_k=1), 'tested_candidates': len(val_ranked), 'limitations': ['Research only; top-one execution approximation.', 'Token universe still selected by all-history coverage.', 'Final test is historical and has been inspected previously; fresh forward data is required.', 'Formula preprocessing must be reproduced by the live runner before deployment.']}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    (out / 'candidates.json').write_text(json.dumps([{'score': s, 'formula': list(f), 'threshold': t} for s, f, t, _ in val_ranked], indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
