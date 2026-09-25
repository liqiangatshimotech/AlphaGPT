"""Offline causal v3 research pipeline: search, freeze, and one final test.

Usage (research only; never instantiates the trading runner or sends orders):

    .venv/bin/python -m model_core.v3_pipeline smoke --out logs/research-v3-smoke
    .venv/bin/python -m model_core.v3_pipeline search --out logs/research-v3-01 \
        --seed 7 --batch 128 --steps 40
    .venv/bin/python -m model_core.v3_pipeline final-test --out logs/research-v3-01

``search`` uses only the training window for rewards and the validation
window for choosing one candidate, then writes ``selection.json``. The test
window is not simulated until ``final-test``; that command records that the
test was consumed and refuses to run again unless ``--acknowledge-rerun`` is
passed (the report then states the test is no longer unseen).

A candidate artifact is written only if every pre-registered criterion passes,
and only inside the run directory. Live strategy files are never written. Any
final-test run that does not pass revokes a candidate left by an earlier run.

The acceptance criteria, validation gate, baselines and stress parameters are
frozen into ``selection.json``; ``final-test`` evaluates the frozen copy and
refuses to run if the code's rules have changed since (that needs a new run).
Every final test appends its window to a ledger shared by sibling run
directories; a later test window overlapping a consumed one (same data
source) can never pass, whatever directory or code it runs from.
"""

import argparse
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

from .causal_v3 import CausalV3FeatureEngineer, CausalV3StackVM
from .v3_artifact import build_candidate, runtime_versions, validate_v3_formula, write_json_atomic
from .v3_data import ResearchDataset, iso, load_from_database, synthetic_dataset, train_end_for_range
from .v3_execution import (
    CompactSeries, CostModel, ExecutionSettings, ExitPolicy, cost_models_from_quotes,
    simulate_portfolio, trade_labels,
)
from .v3_search import FormulaEvaluator, QualityCriteria, rank_candidates, run_search

# Pre-registered simple baselines (fixed before any search in this pipeline).
BASELINE_FORMULAS = {
    "momentum_ret": [0],
    "reversal_neg_ret": [0, 10],
    "pressure": [2],
    "volume_level": [5],
}

# Pre-registered final-test acceptance criteria.
ACCEPTANCE = {
    "validation_gate_passed": True,
    "test_quality_passed": True,
    "test_min_entries": 30,
    "test_min_distinct_mints": 10,
    "test_net_return_baseline_gt": 0.0,
    "test_net_return_adverse_gt": 0.0,
    "test_net_return_latency_900s_gt": 0.0,
    "test_net_return_without_top_mint_gt": 0.0,
    "test_max_drawdown_lt": 0.20,
    "test_min_fill_rate": 0.90,
    "test_beats_all_baselines": True,
}
VALIDATION_GATE = {"min_entries": 10, "net_return_gt": 0.0, "quality_passed": True}
STRESS_LATENCY_SECONDS = 900   # live runner syncs OHLCV every 15 minutes
STRESS_ROUTE_FAILURE_RATE = 0.05
LEDGER_NAME = "research-v3-consumed-test-windows.jsonl"
DEFAULT_LEDGER = Path(__file__).resolve().parents[1] / "logs" / LEDGER_NAME
CANDIDATE_NAME = "candidate_causal_v3.json"


def preregistered_rules():
    """Everything that decides pass/fail, frozen with the selection."""
    return {
        "validation_gate": dict(VALIDATION_GATE),
        "acceptance_criteria": dict(ACCEPTANCE),
        "baseline_formulas": {name: list(f) for name, f in BASELINE_FORMULAS.items()},
        "stress": {"latency_seconds": STRESS_LATENCY_SECONDS, "route_failure_rate": STRESS_ROUTE_FAILURE_RATE},
    }


def rules_digest(rules):
    return hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()[:16]
PROTECTED = ("best_meme_strategy.json", "candidate_meme_strategy.json")


@dataclass
class ResearchConfig:
    out: str
    seed: int = 7
    start: str | None = None
    end: str | None = None
    max_tokens: int | None = None
    batch: int = 128
    steps: int = 40
    max_len: int = 8
    top_k: int = 20
    lr: float = 1e-3
    entropy_coef: float = 0.01
    time_budget_seconds: float | None = None
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    embargo_seconds: int = 5 * 3600
    quote_path: str = "logs/quote_snapshots.jsonl"
    snapshot: str | None = None
    synthetic: bool = False
    synthetic_tokens: int = 20
    synthetic_minutes: int = 4000
    consumed_ledger: str | None = None   # default: DEFAULT_LEDGER (project-level, not per --out)
    settings: dict = field(default_factory=lambda: ExecutionSettings().to_dict())
    policy: dict = field(default_factory=lambda: ExitPolicy().to_dict())
    criteria: dict = field(default_factory=lambda: QualityCriteria().to_dict())

    def digest(self):
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("out", "snapshot", "time_budget_seconds", "consumed_ledger")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _log(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}", flush=True)


def _parse_time(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed


def compute_splits(times, config, split_range=None):
    """Train/validation/test windows over ``split_range`` (the range frozen
    at load time) or, for datasets without one, the first/last candle."""
    if split_range is not None:
        start, end = int(split_range[0]), int(split_range[1])
    else:
        start = int(times[0])
        end = int(times[-1]) + 60
    span = end - start
    train_end = train_end_for_range(start, end, config.train_fraction)
    validation_start = train_end + config.embargo_seconds
    validation_end = start + int(span * (config.train_fraction + config.validation_fraction))
    test_start = validation_end + config.embargo_seconds
    if not (start < train_end < validation_start < validation_end < test_start < end):
        raise ValueError("time range too short for train/validation/test splits with embargo")
    return {
        "train": (start, train_end),
        "validation": (validation_start, validation_end),
        "test": (test_start, end),
    }


def ledger_path(config):
    """Consumed-test-window ledger for the project, independent of ``--out``.

    Defaults to ``<repo>/logs/research-v3-consumed-test-windows.jsonl`` so
    moving or regrouping run directories cannot start a fresh history. The
    resolved path is frozen into ``selection.json`` and used by final-test.
    """
    if config.consumed_ledger:
        return Path(config.consumed_ledger).resolve()
    return Path(DEFAULT_LEDGER).resolve()


def _data_source(dataset):
    return str(dataset.meta.get("source", "unknown"))


def _read_ledger(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _overlapping(entries, source, window, exclude_run_id=None):
    start, end = window
    return [e for e in entries
            if e.get("source") == source and e["start"] < end and start < e["end_exclusive"]
            and not (exclude_run_id and e.get("run_id") == exclude_run_id)]


def consumed_overlaps(path, source, window, exclude_run_id=None):
    """Earlier final tests on the same data source whose window overlaps."""
    return _overlapping(_read_ledger(path), source, window, exclude_run_id)


def claim_test_window(path, source, window, *, run_id, out, fingerprint, acknowledge_rerun=False):
    """Atomically check the ledger for overlaps and record this test window.

    Holds an exclusive lock on ``<ledger>.lock`` across read + append, so of
    several concurrent final tests on overlapping windows at most one sees an
    empty history. Returns ``(earlier, own)``: overlapping entries from other
    runs, and this run's own earlier claims (by ``run_id``, else by
    directory). A non-empty ``own`` means this is a rerun whatever the
    ``test_consumed.json`` marker says; without ``acknowledge_rerun`` the
    claim is refused, so a lost marker cannot yield a second first test.
    """
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = _read_ledger(path)
            overlapping = _overlapping(entries, source, window)
            run_path = str(Path(out).resolve())
            if run_id:
                own = [e for e in overlapping if e.get("run_id") == run_id]
            else:   # pre-run_id selections: identify our own claims by directory
                own = [e for e in overlapping if e.get("run") == run_path]
            if own and not acknowledge_rerun:
                raise SystemExit("ledger shows this run already consumed its test window; "
                                 "pass --acknowledge-rerun (the result cannot pass)")
            earlier = [e for e in overlapping if e not in own]
            entry = {"source": source, "start": int(window[0]), "end_exclusive": int(window[1]),
                     "start_iso": iso(window[0]), "end_iso": iso(window[1]), "run": run_path,
                     "run_id": run_id, "data_fingerprint": fingerprint,
                     "consumed_at": datetime.now(timezone.utc).isoformat()}
            with open(path, "a") as handle:
                handle.write(json.dumps(entry, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return earlier, own


def revoke_candidate(out, reason):
    """Move a previously written candidate out of the current-candidate path.

    The revoked copy keeps its content plus a ``revocation`` stamp, which
    ``load_v3_candidate`` refuses, and an audit line is appended.
    """
    path = Path(out) / CANDIDATE_NAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    stamp = datetime.now(timezone.utc)
    data["revocation"] = {"revoked_at": stamp.isoformat(), "reason": reason,
                          "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    target = Path(out) / "revoked" / f"candidate_causal_v3.{stamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    write_json_atomic(target, data)
    path.unlink()
    with open(Path(out) / "artifact_revocations.jsonl", "a") as handle:
        handle.write(json.dumps({"revoked_at": stamp.isoformat(), "reason": reason, "moved_to": str(target),
                                 "original_sha256": data["revocation"]["original_sha256"]}) + "\n")
    return target


def load_dataset(config, out):
    if config.snapshot:
        dataset = ResearchDataset.from_snapshot(torch.load(config.snapshot, weights_only=False))
    elif config.synthetic:
        dataset = synthetic_dataset(tokens=config.synthetic_tokens, minutes=config.synthetic_minutes, seed=config.seed)
    else:
        dataset = load_from_database(
            start=_parse_time(config.start), end=_parse_time(config.end),
            max_tokens=config.max_tokens, train_fraction=config.train_fraction,
        )
    try:
        dataset.check_consistency()
    except ValueError as exc:
        raise SystemExit(f"dataset is inconsistent: {exc}")
    path = out / "data_snapshot.pt"
    if not config.snapshot:
        torch.save(dataset.to_snapshot(), path)
    return dataset


def data_summary(dataset, splits):
    observed = dataset.observed.cpu().numpy()
    summary = {
        "tokens": dataset.num_tokens,
        "grid_columns": int(len(dataset.times)),
        "first_candle": iso(dataset.times[0]),
        "last_candle": iso(dataset.times[-1]),
        "observed_candles": int(observed.sum()),
        "observed_fraction_of_grid": float(observed.mean()),
        "sol_usd_candles": int(len(dataset.sol_times)),
        "sol_usd_range": [iso(dataset.sol_times[0]), iso(dataset.sol_times[-1])] if len(dataset.sol_times) else None,
        "fingerprint": dataset.fingerprint(),
        "meta": dataset.meta,
        "splits": {},
    }
    for name, (start, end) in splits.items():
        columns = (dataset.times >= start) & (dataset.times < end)
        window = observed[:, columns]
        summary["splits"][name] = {
            "start": iso(start), "end_exclusive": iso(end),
            "grid_columns": int(columns.sum()), "observed_candles": int(window.sum()),
            "active_tokens": int(window.any(axis=1).sum()),
        }
    return summary


class Context:
    """Everything derived deterministically from a dataset and a config."""

    def __init__(self, config, dataset, frozen_costs=None):
        self.config = config
        self.dataset = dataset
        self.settings = ExecutionSettings(**config.settings)
        self.policy = ExitPolicy(**config.policy)
        self.criteria = QualityCriteria(**config.criteria)
        self.costs, self.quote_summary = cost_models_from_quotes(config.quote_path)
        if frozen_costs is not None:
            # The quote log keeps growing; the final test must use the cost
            # scenarios frozen with the selection, not a later recomputation.
            self.costs = {
                name: CostModel(**{k: v for k, v in value.items() if k != "side_fraction"})
                for name, value in frozen_costs.items()
            }
        if dataset.meta.get("max_tokens") is not None and dataset.meta.get("split_range") is None:
            raise SystemExit("token-limited dataset has no frozen split_range; its token-selection cutoff "
                             "cannot be verified - reload it with the current loader")
        self.splits = compute_splits(dataset.times, config, dataset.meta.get("split_range"))
        selection_end = dataset.meta.get("selection_end_epoch")
        if selection_end is not None and int(selection_end) != self.splits["train"][1]:
            raise SystemExit(f"token selection cutoff {iso(selection_end)} differs from the training cutoff "
                             f"{iso(self.splits['train'][1])}; token ranking would read beyond training data")
        self.features = CausalV3FeatureEngineer.compute_features(dataset.raw, dataset.observed)
        self.series = CompactSeries(dataset)
        self.tradable = np.ones(dataset.num_tokens, dtype=bool)
        self.vm = CausalV3StackVM()
        self.observed_np = dataset.observed.cpu().numpy()

    def scores(self, formula):
        raw = self.vm.execute(list(formula), self.features, self.dataset.observed)
        if raw is None:
            return None
        return np.where(self.observed_np, torch.sigmoid(raw.double()).cpu().numpy(), np.nan)

    def quality(self, scores, split):
        from .v3_search import cross_section_quality
        start, end = self.splits[split]
        columns = np.nonzero((self.dataset.times >= start) & (self.dataset.times < end))[0]
        eligible = self.observed_np[:, columns] & self.tradable[:, None]
        return cross_section_quality(scores[:, columns], eligible, self.settings.buy_threshold, self.criteria)

    def simulate(self, scores, split, *, cost="baseline", settings=None, exclude_tokens=()):
        start, end = self.splits[split]
        return simulate_portfolio(
            self.dataset, scores, settings or self.settings, self.policy, self.costs[cost],
            start, end, exclude_tokens=exclude_tokens, tradable=self.tradable,
        )


def _compact(result):
    return {k: v for k, v in result.items() if k not in ("trades", "equity_curve", "decisions", "pnl_by_mint")}


def _no_trade_scores(context):
    return np.where(context.observed_np, 0.0, np.nan)


def command_search(config):
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        raise SystemExit(f"{out}/selection.json exists; use a new --out directory")
    started = time.time()
    dataset = load_dataset(config, out)
    context = Context(config, dataset)
    summary = data_summary(dataset, context.splits)
    write_json_atomic(out / "config.json", {**asdict(config), "config_digest": config.digest(), "versions": runtime_versions()})
    write_json_atomic(out / "data_summary.json", summary)
    _log(f"data: {summary['tokens']} tokens, {summary['observed_candles']} candles, "
         f"{summary['first_candle']}..{summary['last_candle']} fingerprint={summary['fingerprint']}")
    for name, info in summary["splits"].items():
        _log(f"split {name}: {info['start']} -> {info['end_exclusive']} ({info['observed_candles']} candles)")
    ledger = ledger_path(config)
    consumed = consumed_overlaps(ledger, _data_source(dataset), context.splits["test"])
    if consumed:
        _log(f"WARNING: test window overlaps {len(consumed)} previously consumed final test(s); "
             "this run's final test cannot pass")

    train_start, train_end = context.splits["train"]
    labels = trade_labels(context.series, context.policy, context.costs["baseline"], context.settings, train_start, train_end)
    label_status = {str(k): int(v) for k, v in zip(*np.unique(labels["status"], return_counts=True))}
    _log(f"train labels: {label_status} (0 unfilled, 1 resolved, 2 purged, 3 written off)")

    evaluator = FormulaEvaluator(
        dataset, context.features, context.series, labels, context.splits["train"],
        context.settings.buy_threshold, context.criteria, context.tradable,
    )
    history_path = out / "search_history.jsonl"
    history_path.unlink(missing_ok=True)
    history, samples = run_search(
        evaluator, seed=config.seed, batch=config.batch, steps=config.steps, max_len=config.max_len,
        lr=config.lr, entropy_coef=config.entropy_coef, history_path=history_path,
        time_budget_seconds=config.time_budget_seconds, log=_log,
    )
    statuses = {}
    for value in evaluator.cache.values():
        statuses[value["status"]] = statuses.get(value["status"], 0) + 1
    evaluations = [
        {"formula": list(key), **{k: v for k, v in value.items() if k != "quality"},
         "quality_reasons": value.get("quality", {}).get("reasons"),
         "train_quality": value.get("quality")}
        for key, value in sorted(evaluator.cache.items(), key=lambda item: -item[1]["reward"])
    ]
    write_json_atomic(out / "evaluations.json", evaluations)

    candidates = []
    for formula, train in rank_candidates(evaluator.cache, config.top_k):
        scores = context.scores(formula)
        quality = context.quality(scores, "validation")
        result = context.simulate(scores, "validation")
        gate = (quality["passed"] and result["entries"] >= preregistered_rules()["validation_gate"]["min_entries"]
                and result["net_return"] > preregistered_rules()["validation_gate"]["net_return_gt"])
        candidates.append({
            "formula": formula,
            "train": {k: v for k, v in train.items() if k != "quality"},
            "train_quality": train["quality"],
            "validation_quality": quality,
            "validation": _compact(result),
            "validation_gate_passed": bool(gate),
        })
        _log(f"validation {formula}: net={result['net_return']:.2%} dd={result['max_drawdown']:.2%} "
             f"entries={result['entries']} gate={gate}")

    baselines = {}
    for name, formula in preregistered_rules()["baseline_formulas"].items():
        scores = context.scores(formula)
        baselines[name] = {"formula": formula, "validation": _compact(context.simulate(scores, "validation")),
                           "validation_quality": context.quality(scores, "validation")}
    baselines["no_trade"] = {"formula": None, "validation": _compact(context.simulate(_no_trade_scores(context), "validation"))}

    passing = [c for c in candidates if c["validation_gate_passed"]]
    pool = passing or candidates
    selected = max(pool, key=lambda c: (c["validation"]["net_return"], -len(c["formula"]))) if pool else None
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_digest": config.digest(),
        "data_fingerprint": summary["fingerprint"],
        "versions": runtime_versions(),
        "selection_rule": "highest validation net return (baseline cost) among top-k training candidates "
                          "that pass the validation gate; if none pass, the best by validation return is frozen "
                          "for diagnosis and cannot pass acceptance",
        "validation_gate": VALIDATION_GATE,
        "acceptance_criteria": ACCEPTANCE,
        "baseline_formulas": BASELINE_FORMULAS,
        "rules": preregistered_rules(),
        "rules_digest": rules_digest(preregistered_rules()),
        "test_window_previously_consumed": consumed,
        "run_id": uuid.uuid4().hex,
        "splits": {name: [int(a), int(b)] for name, (a, b) in context.splits.items()},
        "consumed_ledger": str(ledger),
        "selected": selected,
        "candidates": candidates,
        "validation_baselines": baselines,
        "search": {
            "method": "AlphaGPT policy gradient with grammar-masked sampling (no LIQ_SCORE, valid postfix only)",
            "seed": config.seed, "batch": config.batch, "steps_requested": config.steps,
            "steps_run": len(history), "samples": samples, "unique_evaluations": len(evaluator.cache),
            "status_counts": statuses, "train_label_status": label_status,
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "costs": {name: model.to_dict() for name, model in context.costs.items()},
        "quote_summary": context.quote_summary,
        "test_evaluated": False,
    }
    write_json_atomic(out / "selection.json", selection)
    write_search_report(out, selection, summary)
    _log(f"search finished: {len(evaluator.cache)} unique formulas; selected "
         f"{selected['formula'] if selected else None}; validation gate "
         f"{'passed' if selected and selected['validation_gate_passed'] else 'NOT passed'}")
    return selection


def _criteria_results(selected, test, baselines, acceptance):
    a = acceptance
    checks = {
        "validation_gate_passed": bool(selected["validation_gate_passed"]),
        "test_quality_passed": bool(test["quality"]["passed"]),
        "test_min_entries": test["baseline"]["entries"] >= a["test_min_entries"],
        "test_min_distinct_mints": test["baseline"]["distinct_mints"] >= a["test_min_distinct_mints"],
        "test_net_return_baseline_gt": test["baseline"]["net_return"] > a["test_net_return_baseline_gt"],
        "test_net_return_adverse_gt": test["adverse_cost"]["net_return"] > a["test_net_return_adverse_gt"],
        "test_net_return_latency_900s_gt": test["latency_900s"]["net_return"] > a["test_net_return_latency_900s_gt"],
        "test_net_return_without_top_mint_gt": test["without_top_mint"]["net_return"] > a["test_net_return_without_top_mint_gt"],
        "test_max_drawdown_lt": test["baseline"]["max_drawdown"] < a["test_max_drawdown_lt"],
        "test_min_fill_rate": (test["baseline"]["fill_rate"] or 0.0) >= a["test_min_fill_rate"],
        "test_beats_all_baselines": all(
            test["baseline"]["net_return"] > b["test"]["net_return"] for b in baselines.values()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}   # never NumPy bools
    unknown = set(a) - set(checks)
    if unknown:
        raise SystemExit(f"frozen acceptance criteria not implemented by this code: {sorted(unknown)}")
    return checks


def _frozen_rules(selection):
    rules = selection.get("rules")
    if rules is None or rules_digest(rules) != selection.get("rules_digest"):
        raise SystemExit("selection has no intact frozen rules; start a new research run")
    if selection["rules_digest"] != rules_digest(preregistered_rules()):
        raise SystemExit("pre-registered acceptance/baseline/stress rules changed since the selection was "
                         "frozen; start a new research run instead of reusing the old pre-registration")
    return rules


def command_final_test(out, *, acknowledge_rerun=False):
    """Run the final test holding an exclusive per-run lock for its duration.

    The ``test_consumed.json`` marker is read inside the lock, so concurrent
    final tests on one run cannot both be treated as the first test; the
    second one fails fast instead of waiting and silently becoming a rerun.
    """
    import fcntl

    out = Path(out)
    if not (out / "selection.json").exists():
        raise SystemExit(f"{out}/selection.json not found")
    with open(out / ".final_test.lock", "a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another final test is running for {out}")
        try:
            return _final_test_locked(out, acknowledge_rerun)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _final_test_locked(out, acknowledge_rerun):
    selection_path = out / "selection.json"
    selection = json.loads(selection_path.read_text())
    marker = out / "test_consumed.json"
    rerun = marker.exists()
    if rerun and not acknowledge_rerun:
        raise SystemExit("final test already run for this selection; pass --acknowledge-rerun "
                         "(the report will state the test window is no longer unseen)")
    stored = json.loads((out / "config.json").read_text())
    stored.pop("config_digest", None)
    stored.pop("versions", None)
    config = replace(ResearchConfig(**stored), out=str(out))   # the directory may have moved
    local_snapshot = out / "data_snapshot.pt"
    snapshot_path = local_snapshot if local_snapshot.exists() else config.snapshot
    if config.digest() != selection["config_digest"]:
        raise SystemExit("config changed since selection was frozen")
    if selection["versions"] != runtime_versions():
        raise SystemExit("runtime feature/VM/signal versions differ from the frozen selection")
    rules = _frozen_rules(selection)
    dataset = ResearchDataset.from_snapshot(torch.load(snapshot_path, weights_only=False))
    try:
        dataset.check_consistency()
    except ValueError as exc:
        raise SystemExit(f"data snapshot is inconsistent: {exc}")
    if dataset.fingerprint() != selection["data_fingerprint"]:
        raise SystemExit("data snapshot fingerprint differs from the frozen selection")
    context = Context(config, dataset, frozen_costs=selection["costs"])
    frozen_splits = selection.get("splits")
    current_splits = {name: [int(a), int(b)] for name, (a, b) in context.splits.items()}
    if frozen_splits is not None and frozen_splits != current_splits:
        raise SystemExit(f"train/validation/test boundaries differ from the frozen selection: "
                         f"{current_splits} != {frozen_splits}")
    selected = selection["selected"]
    source = _data_source(dataset)
    # Use the ledger frozen at selection time (older selections: the project
    # default), and claim the window before any test-period simulation.
    ledger = Path(selection.get("consumed_ledger") or ledger_path(config))
    run_id = selection.get("run_id")
    earlier, own_claims = claim_test_window(ledger, source, context.splits["test"], run_id=run_id, out=out,
                                            fingerprint=selection["data_fingerprint"],
                                            acknowledge_rerun=acknowledge_rerun)
    # The ledger is authoritative: an earlier claim by this run makes this a
    # rerun even if the marker was lost. Acknowledgement only permits a
    # diagnostic rerun; it never restores eligibility to pass.
    previous_runs = json.loads(marker.read_text()).get("previous_runs", 0) + 1 if rerun else len(own_claims)
    rerun = rerun or bool(own_claims)
    write_json_atomic(marker, {"consumed_at": datetime.now(timezone.utc).isoformat(),
                               "previous_runs": previous_runs})
    # Whatever this run concludes, an earlier run's artifact is no longer the
    # current result; it is re-issued below only if this run passes.
    revoked = revoke_candidate(out, "superseded by a final-test rerun" if rerun else "superseded by a new final test")
    if selected is None:
        results = {"selected": None, "passed": False, "reason": "search produced no ranked candidate",
                   "revoked_candidate": str(revoked) if revoked else None}
        write_json_atomic(out / "test_results.json", results)
        write_final_report(out, selection, results, rerun)
        return results

    formula = validate_v3_formula(selected["formula"])
    scores = context.scores(formula)
    test = {"quality": context.quality(scores, "test")}
    base = context.simulate(scores, "test")
    test["baseline"] = base
    test["adverse_cost"] = context.simulate(scores, "test", cost="adverse")
    test["severe_cost"] = context.simulate(scores, "test", cost="severe")
    stress = rules["stress"]
    test["latency_900s"] = context.simulate(scores, "test", settings=replace(context.settings, decision_latency_seconds=stress["latency_seconds"]))
    test["route_failure_5pct"] = context.simulate(scores, "test", settings=replace(context.settings, route_failure_rate=stress["route_failure_rate"]))
    top = base["top_mint"]
    exclude = [dataset.addresses.index(top)] if top and (base["top_mint_pnl_sol"] or 0) > 0 else []
    test["without_top_mint"] = context.simulate(scores, "test", exclude_tokens=exclude)
    test["without_top_mint"]["excluded_mint"] = top if exclude else None

    baselines = {}
    for name, baseline_formula in rules["baseline_formulas"].items():
        baseline_scores = context.scores(baseline_formula)
        baselines[name] = {
            "formula": baseline_formula,
            "test": _compact(context.simulate(baseline_scores, "test")),
            "test_adverse_cost": _compact(context.simulate(baseline_scores, "test", cost="adverse")),
        }
    baselines["no_trade"] = {"formula": None, "test": _compact(context.simulate(_no_trade_scores(context), "test"))}
    checks = _criteria_results(selected, test, baselines, rules["acceptance_criteria"])
    passed = all(checks.values()) and not rerun and not earlier
    results = {
        "selected": formula,
        "checks": checks,
        "passed": passed,
        "test_rerun": rerun,
        "test_window_previously_consumed": earlier,
        "rules_digest": selection["rules_digest"],
        "revoked_candidate": str(revoked) if revoked else None,
        "evidence_level": "exploratory historical backtest (not forward-validated)",
        "test": {name: (_compact(value) if isinstance(value, dict) and "trades" in value else value) for name, value in test.items()},
        "test_trades": base["trades"],
        "test_pnl_by_mint": base["pnl_by_mint"],
        "baselines": baselines,
    }
    write_json_atomic(out / "test_results.json", results)
    if passed:
        artifact = build_candidate(
            formula,
            signal={"buy_threshold": context.settings.buy_threshold, "score": "sigmoid(raw)", "ai_exit_threshold": None},
            exit_policy=context.policy.to_dict(),
            training={"config": asdict(config), "config_digest": config.digest(), "data_fingerprint": selection["data_fingerprint"],
                      "search": selection["search"]},
            selection={"validation": selected["validation"], "test_checks": checks},
        )
        write_json_atomic(out / CANDIDATE_NAME, artifact)
    write_final_report(out, selection, results, rerun)
    _log(f"final test {'PASSED' if passed else 'FAILED'}: " + ", ".join(k for k, v in checks.items() if not v))
    return results


def _pct(value):
    return "—" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.2%}"


def _formula_text(formula):
    from .vocab import FORMULA_VOCAB
    return " ".join(FORMULA_VOCAB.token_names[t] for t in formula) if formula else "—"


def write_search_report(out, selection, summary):
    lines = ["# 因果 v3 搜索与验证集筛选", "",
             f"- 数据指纹 `{summary['fingerprint']}`；{summary['tokens']} 个币；{summary['observed_candles']} 根实际 K 线；"
             f"{summary['first_candle']} 至 {summary['last_candle']}。",
             f"- 搜索：{selection['search']['method']}；采样 {selection['search']['samples']} 次，"
             f"唯一公式 {selection['search']['unique_evaluations']} 个；状态 {selection['search']['status_counts']}。",
             "", "| 公式 | 训练交易 | 训练均值 | t | 验证净收益 | 验证回撤 | 验证入场 | 验证门槛 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in selection["candidates"]:
        lines.append(f"| `{_formula_text(c['formula'])}` | {c['train']['trades']} | {_pct(c['train']['mean_net'])} | "
                     f"{c['train'].get('t_stat', 0):.2f} | {_pct(c['validation']['net_return'])} | "
                     f"{_pct(c['validation']['max_drawdown'])} | {c['validation']['entries']} | {c['validation_gate_passed']} |")
    selected = selection["selected"]
    lines += ["", f"冻结候选：`{_formula_text(selected['formula']) if selected else '无'}`" +
              (f"（验证门槛 {'通过' if selected['validation_gate_passed'] else '未通过'}）" if selected else ""), ""]
    (out / "search_report.md").write_text("\n".join(lines) + "\n")


def write_final_report(out, selection, results, rerun):
    lines = ["# 因果 v3 最终测试", ""]
    if rerun:
        lines += ["**注意：测试段已被运行过，本次为重跑，测试段不再是未见数据，结果不能作为验收依据。**", ""]
    for entry in results.get("test_window_previously_consumed") or []:
        lines += [f"**注意：测试段与此前已消费的最终测试窗口重叠（{entry['start_iso']} → {entry['end_iso']}，"
                  f"`{entry['run']}`），不是未见数据，结论强制为不通过。**", ""]
    if results.get("revoked_candidate"):
        lines += [f"此前生成的候选产物已作废并移至 `{results['revoked_candidate']}`，不可再作为当前合格候选。", ""]
    if results.get("selected") is None:
        lines += ["搜索没有产生可排名候选，无最终测试。", ""]
    else:
        lines += [f"候选：`{_formula_text(results['selected'])}` — 结论：**{'通过' if results['passed'] else '未通过'}**"
                  f"（证据等级：{results['evidence_level']}）", "",
                  "| 情景 | 净收益 | 最大回撤 | 入场 | mint | 成交率 |", "|---|---:|---:|---:|---:|---:|"]
        for name, value in results["test"].items():
            if name == "quality":
                continue
            lines.append(f"| {name} | {_pct(value['net_return'])} | {_pct(value['max_drawdown'])} | {value['entries']} | "
                         f"{value['distinct_mints']} | {_pct(value['fill_rate'])} |")
        lines += ["", "| 基线 | 净收益（基准成本） | 入场 |", "|---|---:|---:|"]
        for name, value in results["baselines"].items():
            lines.append(f"| {name} | {_pct(value['test']['net_return'])} | {value['test']['entries']} |")
        base = results["test"]["baseline"]
        lines += ["", "基准情景分日盈亏（SOL，按入场 UTC 日期）：" + ", ".join(
            f"{day} {pnl:+.3f}" for day, pnl in base["pnl_by_day"].items()) if base["pnl_by_day"] else "基准情景无成交。",
            f"最赚钱 mint：{base['top_mint'] or '—'}（{base['top_mint_pnl_sol'] or 0:+.3f} SOL，"
            f"占正盈利 {_pct(base['top_mint_share_of_positive_pnl'])}）；"
            f"移除后净收益 {_pct(results['test']['without_top_mint']['net_return'])}。",
            f"计数：{base['counters']}"]
        lines += ["", "| 预先固定标准 | 结果 |", "|---|---|"]
        lines += [f"| {k} | {'✓' if v else '✗'} |" for k, v in results["checks"].items()]
        lines += ["", f"候选产物：{'`candidate_causal_v3.json`（仅研究目录，未上线）' if results['passed'] else '未生成'}", ""]
    (out / "final_report.md").write_text("\n".join(lines) + "\n")


def command_smoke(out, seed):
    """Synthetic end-to-end run with a tiny budget plus a timing estimate."""
    config = ResearchConfig(out=out, seed=seed, synthetic=True, batch=16, steps=2, top_k=3,
                            consumed_ledger=str(Path(out) / "smoke_consumed_windows.jsonl"))
    out_path = Path(out)
    if out_path.exists():
        for name in ("selection.json", "test_consumed.json", "test_results.json", CANDIDATE_NAME,
                     "smoke_consumed_windows.jsonl"):
            (out_path / name).unlink(missing_ok=True)
    started = time.time()
    command_search(config)
    command_final_test(out)
    _log(f"smoke run completed in {time.time() - started:.1f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("--out", required=True)
    for name, kind in (("seed", int), ("batch", int), ("steps", int), ("max-len", int), ("top-k", int),
                       ("max-tokens", int), ("time-budget-seconds", float), ("start", str), ("end", str),
                       ("snapshot", str), ("quote-path", str), ("consumed-ledger", str)):
        search.add_argument(f"--{name}", type=kind)
    search.add_argument("--synthetic", action="store_true")
    final = sub.add_parser("final-test")
    final.add_argument("--out", required=True)
    final.add_argument("--acknowledge-rerun", action="store_true")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--out", default="logs/research-v3-smoke")
    smoke.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for protected in PROTECTED:
        if Path(args.out).name == protected:
            raise SystemExit("refusing to use a live strategy file name as output")
    if args.command == "search":
        values = {k: v for k, v in vars(args).items() if k not in ("command",) and v is not None}
        values["synthetic"] = bool(args.synthetic)
        command_search(ResearchConfig(**values))
    elif args.command == "final-test":
        command_final_test(args.out, acknowledge_rerun=args.acknowledge_rerun)
    else:
        command_smoke(args.out, args.seed)


if __name__ == "__main__":
    sys.exit(main())
