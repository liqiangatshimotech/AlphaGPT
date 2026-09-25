"""Artifact versioning, search isolation and full-path causality for v3."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from model_core import v3_pipeline
from model_core.alphagpt import AlphaGPT
from model_core.causal_v3 import CausalV3FeatureEngineer, CausalV3StackVM
from model_core.v3_artifact import (
    build_candidate, load_v3_candidate, validate_v3_formula, write_json_atomic,
)
from model_core.v3_data import ResearchDataset, synthetic_dataset
from model_core.v3_execution import (
    CompactSeries, CostModel, ExecutionSettings, ExitPolicy, simulate_portfolio, trade_labels,
)
from model_core.v3_pipeline import ResearchConfig, compute_splits
from model_core.v3_search import (
    SIZE, STOP, FormulaEvaluator, QualityCriteria, cross_section_quality, grammar_table,
    random_formula, sample_formulas,
)
from model_core.vocab import FORMULA_VOCAB, load_formula

LIVE_FORMULA = [1, 4, 2, 14, 17, 12, 8, 14, 9, 1, 9, 11]

_MODULE_LEDGER = {}


def setUpModule():
    # Never let tests touch the project's real consumed-window ledger.
    _MODULE_LEDGER["dir"] = tempfile.TemporaryDirectory()
    _MODULE_LEDGER["patch"] = mock.patch.object(
        v3_pipeline, "DEFAULT_LEDGER", Path(_MODULE_LEDGER["dir"].name) / "module-ledger.jsonl")
    _MODULE_LEDGER["patch"].start()


def tearDownModule():
    _MODULE_LEDGER["patch"].stop()
    _MODULE_LEDGER["dir"].cleanup()


class IsolatedLedgerCase(unittest.TestCase):
    """Each test gets its own project ledger (synthetic windows coincide)."""

    def setUp(self):
        self._ledger_dir = tempfile.TemporaryDirectory()
        self.ledger = Path(self._ledger_dir.name) / "project" / "ledger.jsonl"
        patcher = mock.patch.object(v3_pipeline, "DEFAULT_LEDGER", self.ledger)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._ledger_dir.cleanup)


def candidate(formula=(0, 10)):
    return build_candidate(
        list(formula),
        signal={"buy_threshold": 0.85, "ai_exit_threshold": None},
        exit_policy=ExitPolicy().to_dict(), training={}, selection={},
    )


class ArtifactVersionTests(unittest.TestCase):
    def test_v3_candidate_round_trips(self):
        data = candidate()
        self.assertEqual(load_v3_candidate(json.loads(json.dumps(data)))["formula"], [0, 10])

    def test_live_loader_rejects_v3_candidates(self):
        with self.assertRaisesRegex(ValueError, "Unsupported strategy vocabulary version"):
            load_formula(candidate())

    def test_v3_loader_rejects_v2_and_mismatched_versions(self):
        with self.assertRaises(ValueError):
            load_v3_candidate([0, 10])
        with self.assertRaises(ValueError):
            load_v3_candidate({"formula": [0, 10], "vocab_version": 2,
                               "token_names": list(FORMULA_VOCAB.token_names)})
        for key, value in (("feature_version", "causal_v3.features.0"), ("vm_version", "v2"),
                           ("exit_policy_version", "ai_exit"), ("vocab_version", 2),
                           ("token_names", ["X"])):
            with self.subTest(key=key):
                data = candidate()
                data[key] = value
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_v3_candidate(data)
        data = candidate()
        data["signal"]["ai_exit_threshold"] = 0.45
        with self.assertRaisesRegex(ValueError, "AI exit"):
            load_v3_candidate(data)

    def test_disabled_and_malformed_formulas_rejected(self):
        for formula in (LIVE_FORMULA, [1], [0, 6], [0, 2], []):
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                validate_v3_formula(formula)

    def test_refuses_live_strategy_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("best_meme_strategy.json", "candidate_meme_strategy.json"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "live strategy"):
                    write_json_atomic(Path(tmp) / name, candidate())
            self.assertEqual(os.listdir(tmp), [])


class GrammarTests(unittest.TestCase):
    def test_random_and_policy_samples_are_valid(self):
        rng = np.random.default_rng(0)
        table = grammar_table(8)
        for _ in range(500):
            formula = random_formula(rng, 8, table)
            self.assertLessEqual(len(formula), 8)
            validate_v3_formula(formula)
        torch.manual_seed(0)
        model = AlphaGPT(input_vocab_size=SIZE + 2, output_size=SIZE + 1, max_len=8)
        formulas, log_prob, _ = sample_formulas(model, 64, 8, torch.from_numpy(table))
        self.assertTrue(torch.isfinite(log_prob).all())
        for formula in formulas:
            self.assertNotIn(STOP, formula)
            validate_v3_formula(formula)

    def test_default_alphagpt_shape_is_unchanged(self):
        model = AlphaGPT()
        logits, _, _ = model(torch.zeros((2, 1), dtype=torch.long))
        self.assertEqual(logits.shape, (2, FORMULA_VOCAB.size))


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.eligible = np.ones((10, 50), dtype=bool)
        self.criteria = QualityCriteria()

    def check(self, scores):
        return cross_section_quality(scores, self.eligible, 0.85, self.criteria)

    def test_saturated_or_broad_or_silent_scores_fail(self):
        rng = np.random.default_rng(0)
        saturated = np.ones((10, 50))
        saturated[0] = 0.2
        self.assertIn("scores saturate", self.check(saturated)["reasons"])
        broad = rng.uniform(0.86, 0.99, (10, 50))
        self.assertIn("buys more than half the universe too often", self.check(broad)["reasons"])
        silent = rng.uniform(0.1, 0.8, (10, 50))
        self.assertIn("buy threshold never reached", self.check(silent)["reasons"])
        nonfinite = silent.copy()
        nonfinite[3, 7] = np.nan
        self.assertIn("non-finite scores", self.check(nonfinite)["reasons"])

    def test_sparse_separable_signal_passes_every_section(self):
        rng = np.random.default_rng(1)
        scores = rng.uniform(0.1, 0.8, (10, 50))
        scores[rng.integers(0, 10, 50), np.arange(50)] = 0.9 + rng.uniform(0, 0.05, 50)
        stats = self.check(scores)
        self.assertTrue(stats["passed"], stats["reasons"])
        self.assertEqual(stats["sections"], 50)


def _truncate(dataset, end_column):
    return ResearchDataset(
        addresses=dataset.addresses, times=dataset.times[:end_column],
        observed=dataset.observed[:, :end_column],
        raw={k: v[:, :end_column] for k, v in dataset.raw.items()},
        prices={k: v[:, :end_column].copy() for k, v in dataset.prices.items()},
        sol_times=dataset.sol_times, sol_close=dataset.sol_close,
    )


class FullPathCausalityTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_dataset(tokens=10, minutes=2500, seed=11, gap_probability=0.08)
        self.vm = CausalV3StackVM()
        self.formulas = ([0, 10], [4, 14, 2, 6], [3, 5, 9, 17], [0, 2, 3, 13, 15], [5, 16, 5, 7])

    def scores(self, dataset, formula):
        features = CausalV3FeatureEngineer.compute_features(dataset.raw, dataset.observed)
        raw = self.vm.execute(formula, features, dataset.observed)
        observed = dataset.observed.numpy()
        return np.where(observed, torch.sigmoid(raw.double()).numpy(), np.nan)

    def test_prefix_scores_and_decisions_do_not_depend_on_later_data(self):
        cut = 1600
        altered = synthetic_dataset(tokens=10, minutes=2500, seed=11, gap_probability=0.08)
        rng = np.random.default_rng(3)
        # Rewrite and delete candles after the cut, including whole columns.
        factor = torch.from_numpy(rng.uniform(0.2, 5.0, (10, len(altered.times) - cut))).float()
        for name in ("open", "high", "low", "close"):
            altered.raw[name][:, cut:] *= factor
        altered.prices["open"][:, cut:] = altered.raw["open"][:, cut:].double().numpy()
        altered.prices["close"][:, cut:] = altered.raw["close"][:, cut:].double().numpy()
        altered.observed[:, cut + 5: cut + 40] = False
        cut_time = int(self.data.times[cut - 1])
        settings = ExecutionSettings(max_buy_fraction=1.0)
        policy = ExitPolicy(max_hold_seconds=3600)
        cost = CostModel("c", 20.0, 10.0, 0.0002)
        for formula in self.formulas:
            with self.subTest(formula=formula):
                full = self.scores(self.data, formula)
                changed = self.scores(altered, formula)
                prefix = self.scores(_truncate(self.data, cut), formula)
                np.testing.assert_allclose(changed[:, :cut], full[:, :cut], rtol=1e-6, atol=1e-7, equal_nan=True)
                np.testing.assert_allclose(prefix, full[:, :cut], rtol=1e-6, atol=1e-7, equal_nan=True)
                start, end = int(self.data.times[0]), int(self.data.times[-1]) + 60
                a = simulate_portfolio(self.data, full, settings, policy, cost, start, end)["decisions"]
                b = simulate_portfolio(altered, changed, settings, policy, cost, start, end)["decisions"]
                early = lambda events: [e for e in events if e[0] <= cut_time + 60]
                self.assertEqual(early(a), early(b))

    def test_training_rewards_ignore_validation_and_test_data(self):
        config = ResearchConfig(out="unused")
        splits = compute_splits(self.data.times, config)
        altered = synthetic_dataset(tokens=10, minutes=2500, seed=11, gap_probability=0.08)
        late = altered.times >= splits["train"][1]
        for name in ("open", "high", "low", "close"):
            altered.raw[name][:, torch.from_numpy(late)] *= 4.0
        altered.prices["open"][:, late] *= 4.0
        altered.prices["close"][:, late] *= 4.0
        rewards = []
        for dataset in (self.data, altered):
            features = CausalV3FeatureEngineer.compute_features(dataset.raw, dataset.observed)
            series = CompactSeries(dataset)
            labels = trade_labels(series, ExitPolicy(max_hold_seconds=3600), CostModel("c", 20, 10, 0.0),
                                  ExecutionSettings(), *splits["train"])
            evaluator = FormulaEvaluator(dataset, features, series, labels, splits["train"], 0.85,
                                         QualityCriteria(min_trades=1, min_mints=1), np.ones(10, dtype=bool))
            rewards.append([evaluator.evaluate(f) for f in self.formulas])
        for before, after in zip(*rewards):
            self.assertEqual(before["reward"], after["reward"])
            self.assertEqual(before.get("trades"), after.get("trades"))

    def test_splits_are_ordered_with_embargo(self):
        config = ResearchConfig(out="unused", embargo_seconds=3600)
        splits = compute_splits(self.data.times, config)
        self.assertLessEqual(splits["train"][1] + 3600, splits["validation"][0])
        self.assertLessEqual(splits["validation"][1] + 3600, splits["test"][0])

    def test_nonfinite_formula_is_invalid_not_repaired(self):
        features = CausalV3FeatureEngineer.compute_features(self.data.raw, self.data.observed)
        features[0, 0, 100] = float("inf")
        self.assertIsNone(self.vm.execute([0, 10], features, self.data.observed))


class PipelineSafetyTests(IsolatedLedgerCase):
    def test_failed_run_writes_no_candidate_and_test_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("best_meme_strategy.json").write_text('{"formula": [0]}')
                out = Path(tmp) / "run"
                config = ResearchConfig(out=str(out), synthetic=True, synthetic_tokens=8, synthetic_minutes=2500,
                                        batch=8, steps=1, top_k=2, quote_path="missing.jsonl")
                with mock.patch.object(v3_pipeline, "_log"):
                    v3_pipeline.command_search(config)
                    with self.assertRaises(SystemExit):
                        v3_pipeline.command_search(config)   # frozen selection is not overwritten
                    results = v3_pipeline.command_final_test(out)
                    self.assertFalse(results["passed"])
                    self.assertFalse((out / "candidate_causal_v3.json").exists())
                    with self.assertRaisesRegex(SystemExit, "already run"):
                        v3_pipeline.command_final_test(out)
                    rerun = v3_pipeline.command_final_test(out, acknowledge_rerun=True)
                    self.assertTrue(rerun["test_rerun"])
                    self.assertIn("不再是未见数据", (out / "final_report.md").read_text())
                self.assertEqual(Path("best_meme_strategy.json").read_text(), '{"formula": [0]}')
                self.assertFalse(Path("candidate_meme_strategy.json").exists())
            finally:
                os.chdir(cwd)

    def test_final_test_uses_costs_frozen_at_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            quotes = Path(tmp) / "quotes.jsonl"
            quotes.write_text(json.dumps({"status": "available", "round_trip_cost_bps": 40.0}) + "\n")
            config = ResearchConfig(out=str(out), synthetic=True, synthetic_tokens=6, synthetic_minutes=2500,
                                    batch=4, steps=1, top_k=1, quote_path=str(quotes))
            with mock.patch.object(v3_pipeline, "_log"):
                selection = v3_pipeline.command_search(config)
                quotes.write_text(json.dumps({"status": "available", "round_trip_cost_bps": 900.0}) + "\n")
                captured = {}
                original = v3_pipeline.Context.__init__

                def spy(context, *args, **kwargs):
                    original(context, *args, **kwargs)
                    captured["baseline"] = context.costs["baseline"].quote_bps_per_side
                with mock.patch.object(v3_pipeline.Context, "__init__", spy):
                    v3_pipeline.command_final_test(out)
            self.assertEqual(captured["baseline"], selection["costs"]["baseline"]["quote_bps_per_side"])
            self.assertEqual(captured["baseline"], 20.0)

    def test_final_test_refuses_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            config = ResearchConfig(out=str(out), synthetic=True, synthetic_tokens=6, synthetic_minutes=2500,
                                    batch=4, steps=1, top_k=1, quote_path="missing.jsonl")
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_search(config)
                stored = json.loads((out / "config.json").read_text())
                stored["settings"]["buy_threshold"] = 0.6
                (out / "config.json").write_text(json.dumps(stored))
                with self.assertRaisesRegex(SystemExit, "config changed"):
                    v3_pipeline.command_final_test(out)
                self.assertFalse((out / "test_consumed.json").exists())


def _small_config(out, **extra):
    values = dict(out=str(out), synthetic=True, synthetic_tokens=6, synthetic_minutes=2500,
                  batch=4, steps=1, top_k=1, quote_path="missing.jsonl")
    values.update(extra)
    return ResearchConfig(**values)


def _all_pass(selected, test, baselines, acceptance):
    return {name: True for name in acceptance}


class FrozenInputsTests(IsolatedLedgerCase):
    def test_fingerprint_covers_every_ohlcv_input(self):
        base = synthetic_dataset(tokens=5, minutes=600, seed=4)
        reference = base.fingerprint()
        self.assertTrue(base.check_consistency())
        for field in ("open", "high", "low", "close", "volume"):
            with self.subTest(field=field):
                data = synthetic_dataset(tokens=5, minutes=600, seed=4)
                data.raw[field][:, -120:] *= 1000.0
                self.assertNotEqual(data.fingerprint(), reference)
        data = synthetic_dataset(tokens=5, minutes=600, seed=4)
        data.sol_close = data.sol_close * 1.01
        self.assertNotEqual(data.fingerprint(), reference)

    def test_consistency_check_rejects_raw_price_mismatch(self):
        data = synthetic_dataset(tokens=5, minutes=600, seed=4)
        observed = data.observed.numpy()
        row, col = np.argwhere(observed)[10]
        data.raw["close"][row, col] *= 2.0
        with self.assertRaisesRegex(ValueError, "raw.close disagrees"):
            data.check_consistency()
        data = synthetic_dataset(tokens=5, minutes=600, seed=4)
        del data.raw["volume"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            data.check_consistency()

    def test_final_test_refuses_changed_feature_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_search(_small_config(out))
                snapshot = torch.load(out / "data_snapshot.pt", weights_only=False)
                snapshot["raw"]["volume"][:, -120:] *= 1000.0
                torch.save(snapshot, out / "data_snapshot.pt")
                with self.assertRaisesRegex(SystemExit, "fingerprint"):
                    v3_pipeline.command_final_test(out)
                snapshot["raw"]["close"][:, -120:] *= 2.0
                torch.save(snapshot, out / "data_snapshot.pt")
                with self.assertRaisesRegex(SystemExit, "inconsistent"):
                    v3_pipeline.command_final_test(out)
            self.assertFalse((out / "test_consumed.json").exists())


class FrozenRulesTests(IsolatedLedgerCase):
    def test_final_test_refuses_rules_changed_after_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                selection = v3_pipeline.command_search(_small_config(out))
                self.assertEqual(selection["rules"]["acceptance_criteria"]["test_min_entries"], 30)
                for patch in (
                    mock.patch.dict(v3_pipeline.ACCEPTANCE, {"test_min_entries": 1}),
                    mock.patch.dict(v3_pipeline.BASELINE_FORMULAS, {"volume_level": [3]}),
                    mock.patch.object(v3_pipeline, "STRESS_LATENCY_SECONDS", 60),
                ):
                    with self.subTest(patch=patch), patch, self.assertRaisesRegex(SystemExit, "rules changed"):
                        v3_pipeline.command_final_test(out)
                self.assertFalse((out / "test_consumed.json").exists())

    def test_acceptance_uses_frozen_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_search(_small_config(out))
                seen = {}

                def spy(selected, test, baselines, acceptance):
                    seen.update(acceptance)
                    return {name: False for name in acceptance}
                selection = json.loads((out / "selection.json").read_text())
                if selection["selected"] is None:
                    self.skipTest("tiny search produced no candidate")
                with mock.patch.object(v3_pipeline, "_criteria_results", spy):
                    v3_pipeline.command_final_test(out)
            self.assertEqual(seen, selection["rules"]["acceptance_criteria"])


class RevocationTests(IsolatedLedgerCase):
    def test_failed_rerun_revokes_earlier_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                selection = v3_pipeline.command_search(_small_config(out))
                if selection["selected"] is None:
                    self.skipTest("tiny search produced no candidate")
                with mock.patch.object(v3_pipeline, "_criteria_results", _all_pass):
                    first = v3_pipeline.command_final_test(out)
                self.assertTrue(first["passed"])
                current = out / "candidate_causal_v3.json"
                self.assertEqual(load_v3_candidate(current)["formula"], selection["selected"]["formula"])
                rerun = v3_pipeline.command_final_test(out, acknowledge_rerun=True)
            self.assertFalse(rerun["passed"])
            self.assertFalse(current.exists())
            with self.assertRaises((FileNotFoundError, ValueError)):
                load_v3_candidate(current)
            revoked = list((out / "revoked").glob("*.json"))
            self.assertEqual(len(revoked), 1)
            with self.assertRaisesRegex(ValueError, "revoked"):
                load_v3_candidate(revoked[0])
            audit = [json.loads(line) for line in (out / "artifact_revocations.jsonl").read_text().splitlines()]
            self.assertEqual(len(audit), 1)
            self.assertIn("作废", (out / "final_report.md").read_text())


class ConsumedWindowTests(IsolatedLedgerCase):
    def test_other_parent_directories_cannot_reclaim_a_consumed_test_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "runs" / "first"
            second = Path(tmp) / "runs" / "group" / "second"
            third = Path(tmp) / "elsewhere" / "deeper" / "third"
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_search(_small_config(first))
                v3_pipeline.command_final_test(first)
                for run, seed in ((second, 8), (third, 9)):
                    with self.subTest(run=str(run)):
                        selection = v3_pipeline.command_search(_small_config(run, seed=seed))
                        self.assertEqual(Path(selection["consumed_ledger"]), self.ledger.resolve())
                        self.assertTrue(selection["test_window_previously_consumed"])
                        if selection["selected"] is None:
                            continue
                        with mock.patch.object(v3_pipeline, "_criteria_results", _all_pass):
                            results = v3_pipeline.command_final_test(run)
                        self.assertFalse(results["passed"])
                        self.assertIn(str(first.resolve()), {e["run"] for e in results["test_window_previously_consumed"]})
                        self.assertFalse((run / "candidate_causal_v3.json").exists())
                        self.assertIn("已消费", (run / "final_report.md").read_text())
            self.assertFalse((Path(tmp) / "runs" / v3_pipeline.LEDGER_NAME).exists())
            self.assertGreaterEqual(len(self.ledger.read_text().splitlines()), 1)

    def test_moved_run_directory_keeps_its_frozen_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "a" / "first", Path(tmp) / "b" / "second"
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_search(_small_config(first))
                v3_pipeline.command_final_test(first)
                selection = v3_pipeline.command_search(_small_config(second, seed=8))
                if selection["selected"] is None:
                    self.skipTest("tiny search produced no candidate")
                moved = Path(tmp) / "archive" / "x" / "second"
                moved.parent.mkdir(parents=True)
                second.rename(moved)
                other = Path(tmp) / "other-default.jsonl"   # a later default must not matter
                with mock.patch.object(v3_pipeline, "DEFAULT_LEDGER", other), \
                        mock.patch.object(v3_pipeline, "_criteria_results", _all_pass):
                    results = v3_pipeline.command_final_test(moved)
            self.assertFalse(results["passed"])
            self.assertTrue(results["test_window_previously_consumed"])
            self.assertFalse(other.exists())

    def test_concurrent_claims_admit_at_most_one_unconsumed_run(self):
        import threading
        import time as time_module

        original = v3_pipeline._read_ledger
        barrier = threading.Barrier(2)

        def slow_read(path):
            entries = original(path)
            time_module.sleep(0.2)   # widen the race window inside the claim
            return entries

        results = {}

        def claim(name):
            barrier.wait()
            results[name], _ = v3_pipeline.claim_test_window(
                self.ledger, "postgres", (100, 200), run_id=name, out=f"/tmp/{name}", fingerprint="f")

        with mock.patch.object(v3_pipeline, "_read_ledger", slow_read):
            threads = [threading.Thread(target=claim, args=(n,)) for n in ("r1", "r2")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        unconsumed = [name for name, earlier in results.items() if not earlier]
        self.assertEqual(len(unconsumed), 1, results)
        self.assertEqual(len(self.ledger.read_text().splitlines()), 2)

    def test_own_rerun_is_not_counted_as_another_run(self):
        claim = lambda window, run_id, **kw: v3_pipeline.claim_test_window(
            self.ledger, kw.pop("source", "s"), window, run_id=run_id, out=f"/tmp/{run_id}", fingerprint="f", **kw)
        self.assertEqual(claim((0, 10), "x"), ([], []))
        with self.assertRaisesRegex(SystemExit, "already consumed"):
            claim((0, 10), "x")
        earlier, own = claim((0, 10), "x", acknowledge_rerun=True)
        self.assertEqual(earlier, [])
        self.assertEqual({e["run_id"] for e in own}, {"x"})
        earlier, own = claim((5, 20), "y")
        self.assertEqual({e["run_id"] for e in earlier}, {"x"})
        self.assertEqual(own, [])
        self.assertEqual(claim((20, 30), "z"), ([], []))
        self.assertEqual(claim((0, 10), "w", source="t"), ([], []))


class SerializationTests(IsolatedLedgerCase):
    def test_numpy_values_round_trip_as_native_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            write_json_atomic(path, {"f": np.bool_(False), "t": np.bool_(True), "i": np.int64(3),
                                     "x": np.float32(0.5), "a": np.array([1, 2]), "p": Path("a/b")})
            data = json.loads(path.read_text())
            self.assertIs(data["f"], False)
            self.assertIs(data["t"], True)
            self.assertEqual((data["i"], data["x"], data["a"], data["p"]), (3, 0.5, [1, 2], "a/b"))
            with self.assertRaises(TypeError):
                write_json_atomic(Path(tmp) / "y.json", {"bad": object()})

    def test_final_test_checks_reload_as_booleans(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                selection = v3_pipeline.command_search(_small_config(out))
                if selection["selected"] is None:
                    self.skipTest("tiny search produced no candidate")
                returned = v3_pipeline.command_final_test(out)
            stored = json.loads((out / "test_results.json").read_text())
            self.assertTrue(stored["checks"])
            for name, value in stored["checks"].items():
                self.assertIs(type(value), bool, name)
                self.assertEqual(value, returned["checks"][name], name)
            self.assertIs(type(stored["passed"]), bool)


class SplitRangeTests(unittest.TestCase):
    def test_token_selection_cutoff_is_the_training_cutoff(self):
        # Review repro: request 10,000 minutes; the selected token A stops at
        # minute 5,000. Splits must follow the frozen request range, so the
        # training cutoff stays at the 6,000-minute selection cutoff.
        from model_core.v3_data import build_dataset, train_end_for_range

        t0 = 1_758_000_000
        rows = [(t0 + 60 * m, "A", 1.0, 1.0, 1.0, 1.0, 1.0) for m in range(1, 5000)]
        sol_rows = [(t0 + 60 * m, 100.0) for m in range(10_000)]
        range_ = [t0, t0 + 60 * 10_000]
        selection_end = train_end_for_range(*range_, 0.6)
        self.assertEqual(selection_end, t0 + 60 * 6000)
        data = build_dataset(rows, sol_rows, {"split_range": range_, "selection_end_epoch": selection_end})
        config = ResearchConfig(out="unused", embargo_seconds=3600)
        splits = compute_splits(data.times, config, data.meta["split_range"])
        self.assertEqual(splits["train"][1], selection_end)
        # The old behaviour (splits from loaded candles) would have moved it.
        self.assertLess(compute_splits(data.times, config)["train"][1], selection_end)
        v3_pipeline.Context(config, data)   # consistent: accepted

    def test_context_refuses_mismatched_selection_cutoff(self):
        from model_core.v3_data import build_dataset

        t0 = 1_758_000_000
        rows = [(t0 + 60 * m, "A", 1.0, 1.0, 1.0, 1.0, 1.0) for m in range(1, 5000)]
        sol_rows = [(t0 + 60 * m, 100.0) for m in range(10_000)]
        data = build_dataset(rows, sol_rows, {"split_range": [t0, t0 + 600_000],
                                              "selection_end_epoch": t0 + 60 * 6001})
        with self.assertRaisesRegex(SystemExit, "selection cutoff"):
            v3_pipeline.Context(ResearchConfig(out="unused", embargo_seconds=3600), data)


class DatabaseLoaderTests(unittest.TestCase):
    def test_token_ranking_ignores_test_and_invalid_rows(self):
        import sqlite3
        from model_core.v3_data import SOL_MINT, candle_query_sql

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE ohlcv (time INTEGER, address TEXT, open REAL, high REAL, low REAL, "
                     "close REAL, volume REAL, source TEXT)")
        rows = []
        for minute in range(100):
            rows.append((minute, "GOOD1", 1, 1, 1, 1, 5, "birdeye"))
            if minute < 60:
                rows.append((minute, "GOOD2", 1, 1, 1, 1, 5, None))
            # Many more rows, all unusable: test source or invalid OHLCV.
            rows.append((minute, "POLLUTED", 1, 1, 1, 1, 5, "test"))
            rows.append((minute + 0.5, "POLLUTED", 1, 1, 1, 1, 5, "test"))
            rows.append((minute, "BROKEN", 0, 1, 1, 1, 5, "birdeye"))
            rows.append((minute + 0.5, "BROKEN", 1, 0.5, 1, 1, 5, "birdeye"))
        conn.executemany("INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        sql = candle_query_sql(limit_tokens=True).replace(":start", "?").replace(":end", "?") \
            .replace(":selection_end", "?").replace(":sol", "?").replace(":limit", "?")
        # Named-parameter order in the query text: start, end, sol, start, selection_end, sol, limit.
        loaded = conn.execute(sql, (0, 1000, SOL_MINT, 0, 1000, SOL_MINT, 2)).fetchall()
        self.assertEqual({row[1] for row in loaded}, {"GOOD1", "GOOD2"})

    def test_standalone_cli_uses_database_settings_from_dotenv(self):
        import subprocess
        import sys

        script = """
import os, sys
from unittest import mock
import model_core.v3_pipeline  # imports ModelConfig before any .env is loaded
from model_core import v3_data

class Stop(Exception):
    pass

def fake_load_dotenv(*args, **kwargs):
    os.environ["DB_HOST"] = "dotenv-host.invalid"
    os.environ["DB_NAME"] = "dotenv_db"
    return True

def fake_engine(url, *args, **kwargs):
    print("URL_HOST", url.split("@", 1)[1])
    raise Stop()

with mock.patch("dotenv.load_dotenv", fake_load_dotenv), mock.patch("sqlalchemy.create_engine", fake_engine):
    try:
        v3_data.load_from_database()
    except Stop:
        pass
"""
        env = {k: v for k, v in os.environ.items() if not k.startswith("DB_")}
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("URL_HOST dotenv-host.invalid:5432/dotenv_db", result.stdout)


class SameRunFinalTestTests(IsolatedLedgerCase):
    def _searched_run(self, tmp):
        out = Path(tmp) / "run"
        with mock.patch.object(v3_pipeline, "_log"):
            selection = v3_pipeline.command_search(_small_config(out))
        if selection["selected"] is None:
            self.skipTest("tiny search produced no candidate")
        return out

    def test_concurrent_final_tests_on_one_run_admit_only_one(self):
        import threading
        import time as time_module

        with tempfile.TemporaryDirectory() as tmp:
            out = self._searched_run(tmp)
            barrier = threading.Barrier(2)
            original_claim = v3_pipeline.claim_test_window

            def slow_claim(*args, **kwargs):
                result = original_claim(*args, **kwargs)
                time_module.sleep(0.5)   # keep the first test inside its critical section
                return result

            outcomes = []

            def run():
                barrier.wait()
                try:
                    outcomes.append(("ok", v3_pipeline.command_final_test(out)))
                except SystemExit as exc:
                    outcomes.append(("exit", str(exc)))

            with mock.patch.object(v3_pipeline, "_log"), \
                    mock.patch.object(v3_pipeline, "claim_test_window", slow_claim), \
                    mock.patch.object(v3_pipeline, "_criteria_results", _all_pass):
                threads = [threading.Thread(target=run) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            kinds = sorted(kind for kind, _ in outcomes)
            self.assertEqual(kinds, ["exit", "ok"], outcomes)
            self.assertIn("another final test is running", next(v for k, v in outcomes if k == "exit"))
            self.assertEqual(len(self.ledger.read_text().splitlines()), 1)
            self.assertEqual(json.loads((out / "test_consumed.json").read_text())["previous_runs"], 0)

    def test_lost_marker_allows_acknowledged_diagnostic_rerun_that_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._searched_run(tmp)
            with mock.patch.object(v3_pipeline, "_log"), \
                    mock.patch.object(v3_pipeline, "_criteria_results", _all_pass):
                first = v3_pipeline.command_final_test(out)
                self.assertTrue(first["passed"])
                self.assertTrue((out / "candidate_causal_v3.json").exists())
                (out / "test_consumed.json").unlink()
                rerun = v3_pipeline.command_final_test(out, acknowledge_rerun=True)
            self.assertTrue(rerun["test_rerun"])
            self.assertFalse(rerun["passed"])
            self.assertEqual(rerun["test_window_previously_consumed"], [])   # own claim is not "another run"
            self.assertFalse((out / "candidate_causal_v3.json").exists())
            self.assertEqual(len(list((out / "revoked").glob("*.json"))), 1)
            self.assertEqual(json.loads((out / "test_consumed.json").read_text())["previous_runs"], 1)
            self.assertIn("不再是未见数据", (out / "final_report.md").read_text())
            self.assertEqual(len(self.ledger.read_text().splitlines()), 2)

    def test_lost_marker_cannot_make_a_second_first_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self._searched_run(tmp)
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_final_test(out)
                (out / "test_consumed.json").unlink()
                with mock.patch.object(v3_pipeline, "_criteria_results", _all_pass), \
                        self.assertRaisesRegex(SystemExit, "already consumed"):
                    v3_pipeline.command_final_test(out)
            self.assertFalse((out / "candidate_causal_v3.json").exists())


class SplitMetadataFreezeTests(IsolatedLedgerCase):
    def test_fingerprint_covers_split_metadata(self):
        data = synthetic_dataset(tokens=4, minutes=600, seed=4)
        plain = data.fingerprint()
        start, end = int(data.times[0]), int(data.times[-1]) + 60
        data.meta.update(split_range=[start, end], selection_end_epoch=start + int((end - start) * 0.6), max_tokens=3)
        frozen = data.fingerprint()
        self.assertNotEqual(frozen, plain)
        data.meta.update(split_range=[start, end - 36000], selection_end_epoch=start + int((end - 36000 - start) * 0.6))
        self.assertNotEqual(data.fingerprint(), frozen)   # both moved in sync: still detected
        for key in ("split_range", "selection_end_epoch", "max_tokens"):
            with self.subTest(removed=key):
                data.meta.update(split_range=[start, end], selection_end_epoch=start + int((end - start) * 0.6),
                                 max_tokens=3)
                self.assertEqual(data.fingerprint(), frozen)
                data.meta[key] = None
                self.assertNotEqual(data.fingerprint(), frozen)

    def test_final_test_refuses_synchronised_split_metadata_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = synthetic_dataset(tokens=6, minutes=2500, seed=7)
            start, end = int(data.times[0]), int(data.times[-1]) + 60
            data.meta.update(split_range=[start, end], max_tokens=6,
                             selection_end_epoch=start + int((end - start) * 0.6))
            snapshot = Path(tmp) / "snapshot.pt"
            torch.save(data.to_snapshot(), snapshot)
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                selection = v3_pipeline.command_search(_small_config(out, synthetic=False, snapshot=str(snapshot)))
                self.assertEqual(selection["splits"]["train"][1], data.meta["selection_end_epoch"])
                edited = torch.load(snapshot, weights_only=False)
                new_end = end - 36000
                edited["meta"]["split_range"] = [start, new_end]
                edited["meta"]["selection_end_epoch"] = start + int((new_end - start) * 0.6)
                torch.save(edited, snapshot)
                with self.assertRaisesRegex(SystemExit, "fingerprint"):
                    v3_pipeline.command_final_test(out)
            self.assertFalse((out / "test_consumed.json").exists())
            self.assertFalse(self.ledger.exists() and self.ledger.read_text().strip())

    def test_final_test_refuses_changed_split_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with mock.patch.object(v3_pipeline, "_log"):
                v3_pipeline.command_search(_small_config(out))
                original = v3_pipeline.compute_splits

                def shifted(*args, **kwargs):
                    splits = original(*args, **kwargs)
                    return {**splits, "test": (splits["test"][0] - 3600, splits["test"][1])}
                with mock.patch.object(v3_pipeline, "compute_splits", shifted), \
                        self.assertRaisesRegex(SystemExit, "boundaries differ"):
                    v3_pipeline.command_final_test(out)
            self.assertFalse((out / "test_consumed.json").exists())

    def test_token_limited_dataset_without_split_range_is_refused(self):
        data = synthetic_dataset(tokens=4, minutes=2500, seed=4)
        data.meta["max_tokens"] = 3
        with self.assertRaisesRegex(SystemExit, "no frozen split_range"):
            v3_pipeline.Context(ResearchConfig(out="unused", embargo_seconds=3600), data)


if __name__ == "__main__":
    unittest.main()
