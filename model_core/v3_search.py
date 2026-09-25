"""Budgeted causal v3 formula search.

Formulas are sampled from AlphaGPT under a grammar mask: every sampled
sequence is a valid postfix expression of at most ``max_len`` tokens and never
contains LIQ_SCORE, so no budget is spent on stack-invalid expressions. An
extra STOP action (not a formula token) lets lengths vary.

Rewards use the *training* window only: executable-rule trade labels for each
signal onset (first observed candle at or above the buy threshold), after
costs, purged where the trade outcome crosses the training end.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .alphagpt import AlphaGPT
from .causal_v3 import CausalV3StackVM
from .ops import OPS_CONFIG
from .v3_artifact import DISABLED_TOKENS
from .vocab import FORMULA_VOCAB

SIZE = FORMULA_VOCAB.size
OFFSET = FORMULA_VOCAB.operator_offset
ARITY = {OFFSET + index: cfg[2] for index, cfg in enumerate(OPS_CONFIG)}
STOP = SIZE             # output action index
BOS_INPUT = SIZE        # input ids: formula tokens, BOS, STOP
STOP_INPUT = SIZE + 1


def grammar_table(max_len):
    """allowed[depth, position, action] for depth 0..max_len+1."""
    allowed = np.zeros((max_len + 2, max_len, SIZE + 1), dtype=bool)
    for depth in range(max_len + 2):
        for position in range(max_len):
            remaining = max_len - position - 1
            for token in range(SIZE):
                if token in DISABLED_TOKENS:
                    continue
                if token < OFFSET:
                    after = depth + 1
                else:
                    if depth < ARITY[token]:
                        continue
                    after = depth + 1 - ARITY[token]
                allowed[depth, position, token] = after >= 1 and after - 1 <= 2 * remaining
            allowed[depth, position, STOP] = depth == 1 and position >= 1
    return allowed


def depth_delta(token):
    return 1 if token < OFFSET else 1 - ARITY[token]


def random_formula(rng, max_len, table=None):
    table = grammar_table(max_len) if table is None else table
    formula, depth = [], 0
    for position in range(max_len):
        choices = np.nonzero(table[depth, position])[0]
        action = int(rng.choice(choices))
        if action == STOP:
            break
        formula.append(action)
        depth += depth_delta(action)
    return formula


@dataclass(frozen=True)
class QualityCriteria:
    max_saturation_rate: float = 0.01
    max_over_half_rate: float = 0.05
    max_no_spread_rate: float = 0.05
    max_tied_buy_rate: float = 0.05
    max_buy_coverage: float = 0.25
    min_trades: int = 30
    min_mints: int = 5

    def to_dict(self):
        return asdict(self)


def cross_section_quality(scores, eligible, threshold, criteria):
    """Checks every decision cross-section (grid column) in the window.

    ``scores``/``eligible`` are [tokens, columns]. Normal "no buy" columns are
    allowed; a window where the signal never buys fails.
    """
    s = np.where(eligible, scores, np.nan)
    n_eligible = eligible.sum(axis=0)
    total = int(eligible.sum())
    nonfinite = int((eligible & ~np.isfinite(scores)).sum())
    sections = n_eligible >= 2
    buys = eligible & (np.nan_to_num(s, nan=-1.0) >= threshold)
    n_buys = buys.sum(axis=0)
    with np.errstate(all="ignore"):
        high = np.nanmax(np.where(eligible, s, -np.inf), axis=0)
        low = np.nanmin(np.where(eligible, s, np.inf), axis=0)
        buy_high = np.max(np.where(buys, s, -np.inf), axis=0)
        buy_low = np.min(np.where(buys, s, np.inf), axis=0)
    n_sections = int(sections.sum())
    saturated = int((eligible & ((s >= 1 - 1e-6) | (s <= 1e-6))).sum())
    stats = {
        "sections": n_sections,
        "mean_eligible": float(n_eligible[sections].mean()) if n_sections else 0.0,
        "observations": total,
        "nonfinite": nonfinite,
        "saturation_rate": saturated / total if total else 1.0,
        "any_buy_rate": float((sections & (n_buys > 0)).sum() / n_sections) if n_sections else 0.0,
        "over_half_rate": float((sections & (n_buys > n_eligible * 0.5)).sum() / n_sections) if n_sections else 1.0,
        "no_spread_rate": float((sections & (high - low < 0.01)).sum() / n_sections) if n_sections else 1.0,
        "tied_buy_rate": float((sections & (n_buys > 1) & (buy_high - buy_low < 0.01)).sum() / n_sections) if n_sections else 0.0,
        "buy_coverage": float(buys.sum() / total) if total else 0.0,
        "score_quantiles": [float(v) for v in np.nanquantile(s[eligible], [0.01, 0.1, 0.5, 0.9, 0.99])] if total else [],
    }
    reasons = []
    if n_sections == 0:
        reasons.append("no cross-section has two eligible tokens")
    if nonfinite:
        reasons.append("non-finite scores")
    if stats["saturation_rate"] > criteria.max_saturation_rate:
        reasons.append("scores saturate")
    if stats["over_half_rate"] > criteria.max_over_half_rate:
        reasons.append("buys more than half the universe too often")
    if stats["no_spread_rate"] > criteria.max_no_spread_rate:
        reasons.append("scores cannot separate candidates too often")
    if stats["tied_buy_rate"] > criteria.max_tied_buy_rate:
        reasons.append("tied buy candidates too often")
    if stats["buy_coverage"] > criteria.max_buy_coverage:
        reasons.append("buy coverage too broad")
    if stats["any_buy_rate"] == 0:
        reasons.append("buy threshold never reached")
    stats["passed"] = not reasons
    stats["reasons"] = reasons
    return stats


class FormulaEvaluator:
    def __init__(self, dataset, features, series, train_labels, train_window, threshold,
                 criteria, tradable):
        self.dataset = dataset
        self.features = features
        self.observed = dataset.observed
        self.observed_np = dataset.observed.cpu().numpy()
        self.series = series
        self.threshold = threshold
        self.criteria = criteria
        self.tradable = tradable
        self.vm = CausalV3StackVM()
        start, end = train_window
        self.train_columns = np.nonzero((dataset.times >= start) & (dataset.times < end))[0]
        self.net = np.full(series.size, np.nan)
        self.net[train_labels["decision_index"]] = train_labels["net"]
        self.train_compact = (series.t >= start) & (series.t < end) & tradable[series.tok]
        self.cache = {}

    def score_grid(self, formula):
        raw = self.vm.execute(list(formula), self.features, self.observed)
        if raw is None:
            return None
        scores = torch.sigmoid(raw.double()).cpu().numpy()
        return np.where(self.observed_np, scores, np.nan)

    def quality(self, scores, columns):
        eligible = self.observed_np[:, columns] & self.tradable[:, None]
        return cross_section_quality(scores[:, columns], eligible, self.threshold, self.criteria)

    def onsets(self, scores):
        compact = scores[self.series.tok, self.series.col]
        signal = np.nan_to_num(compact, nan=-1.0) >= self.threshold
        previous = np.zeros_like(signal)
        previous[1:] = signal[:-1]
        return signal & ~(self.series.has_previous & previous)

    def evaluate(self, formula):
        key = tuple(formula)
        if key in self.cache:
            return self.cache[key]
        scores = self.score_grid(formula)
        if scores is None:
            result = {"reward": -8.0, "status": "invalid"}
        else:
            quality = self.quality(scores, self.train_columns)
            onset = self.onsets(scores) & self.train_compact
            trades = onset & np.isfinite(self.net)
            values = self.net[trades]
            n = int(values.size)
            mints = int(np.unique(self.series.tok[trades]).size)
            signature = hashlib.sha1(np.packbits(onset).tobytes()).hexdigest()[:16]
            result = {
                "status": "ok", "quality": quality, "trades": n, "mints": mints,
                "mean_net": float(values.mean()) if n else None,
                "std_net": float(values.std()) if n > 1 else None,
                "win_rate": float((values > 0).mean()) if n else None,
                "onsets": int(onset.sum()),
                "purged_onsets": int((onset & ~np.isfinite(self.net)).sum()),
                "signature": signature,
            }
            if not quality["passed"]:
                result["reward"] = -7.0
                result["status"] = "quality_failed"
            elif n < self.criteria.min_trades:
                result["reward"] = -6.0 + n / self.criteria.min_trades
                result["status"] = "too_few_trades"
            elif mints < self.criteria.min_mints:
                result["reward"] = -5.5
                result["status"] = "too_few_mints"
            else:
                std = max(result["std_net"] or 0.0, 1e-6)
                result["t_stat"] = result["mean_net"] / (std / math.sqrt(n))
                # Any breadth-passing formula, even a losing one, outranks the
                # failure states so the policy is not pushed toward never trading.
                result["reward"] = float(np.clip(result["t_stat"] / 2.0, -5.0, 5.0))
        self.cache[key] = result
        return result


def sample_formulas(model, batch, max_len, table):
    device = next(model.parameters()).device
    inputs = torch.full((batch, 1), BOS_INPUT, dtype=torch.long, device=device)
    depth = torch.zeros(batch, dtype=torch.long, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    log_prob = torch.zeros(batch, device=device)
    entropy = torch.zeros(batch, device=device)
    actions = []
    for position in range(max_len):
        logits, _, _ = model(inputs)
        mask = table[depth.clamp(max=table.shape[0] - 1), position]
        stop_only = torch.zeros_like(mask)
        stop_only[:, STOP] = True
        mask = torch.where(done[:, None], stop_only, mask)
        dist = Categorical(logits=logits.masked_fill(~mask, float("-inf")))
        action = dist.sample()
        active = ~done
        log_prob = log_prob + torch.where(active, dist.log_prob(action), 0.0)
        entropy = entropy + torch.where(active, dist.entropy(), 0.0)
        is_stop = action == STOP
        delta = torch.tensor([depth_delta(t) if t < SIZE else 0 for t in range(SIZE + 1)], device=device)
        depth = torch.where(active & ~is_stop, depth + delta[action], depth)
        done = done | is_stop
        actions.append(torch.where(active, action, torch.full_like(action, STOP)))
        next_input = torch.where(action == STOP, torch.full_like(action, STOP_INPUT), action)
        inputs = torch.cat([inputs, next_input[:, None]], dim=1)
    rows = torch.stack(actions, dim=1).tolist()
    formulas = [[t for t in row if t != STOP] for row in rows]
    return formulas, log_prob, entropy


def run_search(evaluator, *, seed, batch, steps, max_len, lr=1e-3, entropy_coef=0.01,
               history_path=None, time_budget_seconds=None, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    table = torch.from_numpy(grammar_table(max_len))
    model = AlphaGPT(input_vocab_size=SIZE + 2, output_size=SIZE + 1, max_len=max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    samples = 0
    started = time.time()
    best = (-math.inf, None)
    for step in range(steps):
        formulas, log_prob, entropy = sample_formulas(model, batch, max_len, table)
        samples += len(formulas)
        rewards = torch.tensor([evaluator.evaluate(f)["reward"] for f in formulas], dtype=torch.float32)
        for formula, reward in zip(formulas, rewards.tolist()):
            if reward > best[0]:
                best = (reward, formula)
        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        loss = -(log_prob * advantage).mean() - entropy_coef * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        statuses = {}
        for formula in formulas:
            status = evaluator.cache[tuple(formula)]["status"]
            statuses[status] = statuses.get(status, 0) + 1
        row = {
            "step": step, "samples": samples, "unique_evaluations": len(evaluator.cache),
            "mean_reward": float(rewards.mean()), "max_reward": float(rewards.max()),
            "best_reward": best[0], "best_formula": best[1], "statuses": statuses,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        history.append(row)
        if history_path:
            with open(history_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        log(f"step {step + 1}/{steps} unique={row['unique_evaluations']} mean={row['mean_reward']:.3f} "
            f"best={best[0]:.3f} {best[1]} ({row['elapsed_seconds']}s)")
        if time_budget_seconds and time.time() - started > time_budget_seconds:
            log("time budget reached; stopping search")
            break
    return history, samples


def rank_candidates(cache, top_k):
    """Top distinct-signal formulas that passed training quality and breadth."""
    ranked = sorted(
        ((key, value) for key, value in cache.items() if "t_stat" in value),
        key=lambda item: (-item[1]["reward"], len(item[0]), item[0]),
    )
    chosen, signatures = [], set()
    for key, value in ranked:
        if value["signature"] in signatures:
            continue
        signatures.add(value["signature"])
        chosen.append((list(key), value))
        if len(chosen) >= top_k:
            break
    return chosen
