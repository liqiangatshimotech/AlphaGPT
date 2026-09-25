"""Regression checks for training candidates that saturate live scores."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model_core.backtest import MemeBacktest
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine


class ModelQualityTests(unittest.TestCase):
    def setUp(self):
        self.bt = MemeBacktest()
        self.raw = {"liquidity": torch.full((6, 30), 1_000_000.0)}

    def test_historical_variance_does_not_rescue_latest_saturation(self):
        factors = torch.arange(180.0).reshape(6, 30)
        factors[:, -1] = 1_000_000.0
        self.assertGreater(factors.std().item(), 1e-4)

        valid, reason = self.bt.check_candidate_quality(factors, self.raw)

        self.assertFalse(valid)
        self.assertIn("cannot separate", reason)

    def test_tradable_scores_must_separate_buy_candidates(self):
        factors = torch.zeros(6, 30)
        factors[:, -1] = torch.tensor([3.0, 3.0, 3.0, -3.0, -2.0, -1.0])
        valid, reason = self.bt.check_candidate_quality(factors, self.raw)
        self.assertFalse(valid)
        self.assertIn("buy candidates", reason)

        factors[:, -1] = torch.tensor([3.0, 2.5, -3.0, -2.0, -1.0, 0.0])
        valid, _ = self.bt.check_candidate_quality(factors, self.raw)
        self.assertTrue(valid)

    def test_rejects_nonfinite_and_all_buy_outputs(self):
        factors = torch.full((6, 30), 3.0)
        factors[:, -1] = torch.tensor([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
        valid, reason = self.bt.check_candidate_quality(factors, self.raw)
        self.assertFalse(valid)
        self.assertIn("more than half", reason)

        factors[0, 0] = float("nan")
        valid, reason = self.bt.check_candidate_quality(factors, self.raw)
        self.assertFalse(valid)
        self.assertIn("non-finite", reason)

    def test_only_tradable_tokens_count_toward_score_spread(self):
        factors = torch.full((6, 30), 3.0)
        factors[-1, -1] = -3.0
        self.raw["liquidity"][-1, -1] = 100.0
        valid, reason = self.bt.check_candidate_quality(factors, self.raw)
        self.assertFalse(valid)
        self.assertIn("cannot separate", reason)

    def test_backtest_scores_active_tokens_and_masks_missing_returns(self):
        factors = torch.full((6, 30), -3.0)
        factors[:2] = 3.0
        returns = torch.full((6, 30), 0.03)
        returns[0, 0] = float("-inf")  # An absent pre-listing open.

        score, mean_return = self.bt.evaluate(factors, self.raw, returns)

        self.assertTrue(torch.isfinite(score).item())
        self.assertGreater(score.item(), 0)
        self.assertGreater(mean_return, 0)

    def test_training_and_holdout_windows_do_not_share_labels(self):
        training, holdout = self.bt.temporal_windows(30)
        self.assertEqual(training, slice(0, 20))
        self.assertEqual(holdout, slice(22, 28))

        factors = torch.full((6, 30), -3.0)
        factors[:2] = 3.0
        returns = torch.full((6, 30), 0.03)
        returns[:, holdout] = -0.03
        train_score, _ = self.bt.evaluate(factors, self.raw, returns, period=training)
        holdout_score, _ = self.bt.evaluate(factors, self.raw, returns, period=holdout)

        self.assertGreater(train_score.item(), 0)
        self.assertLess(holdout_score.item(), 0)

    def test_failed_final_validation_preserves_existing_strategy_file(self):
        raw = {
            "open": torch.ones(6, 30),
            "high": torch.ones(6, 30),
            "low": torch.ones(6, 30),
            "close": torch.ones(6, 30),
            "volume": torch.ones(6, 30),
            "liquidity": self.raw["liquidity"],
            "fdv": torch.full((6, 30), 2_000_000.0),
        }
        engine = AlphaEngine.__new__(AlphaEngine)
        engine.use_lord = False
        engine.bt = self.bt
        engine.loader = SimpleNamespace(
            raw_data_cache=raw,
            target_ret=torch.full((6, 30), 0.03),
            feat_tensor=torch.zeros(6, 6, 30),
        )
        engine.vm = SimpleNamespace(
            execute=lambda formula, features: torch.full((6, 30), 1_000_000.0)
        )
        engine.best_formula = [1]
        engine.best_score = 1.0

        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                strategy = Path("best_meme_strategy.json")
                strategy.write_text(json.dumps({"formula": [0]}))
                with patch.object(ModelConfig, "TRAIN_STEPS", 0):
                    with self.assertRaisesRegex(RuntimeError, "latest-score validation"):
                        engine.train()
                self.assertEqual(json.loads(strategy.read_text()), {"formula": [0]})
                self.assertFalse(Path("candidate_meme_strategy.json").exists())
            finally:
                os.chdir(previous_cwd)

    def test_successful_validation_writes_candidate_not_live_strategy(self):
        raw = {
            "open": torch.ones(6, 30),
            "high": torch.ones(6, 30),
            "low": torch.ones(6, 30),
            "close": torch.ones(6, 30),
            "volume": torch.ones(6, 30),
            "liquidity": self.raw["liquidity"],
            "fdv": torch.full((6, 30), 2_000_000.0),
        }
        engine = AlphaEngine.__new__(AlphaEngine)
        engine.use_lord = False
        engine.bt = self.bt
        engine.loader = SimpleNamespace(
            raw_data_cache=raw,
            target_ret=torch.full((6, 30), 0.03),
            feat_tensor=torch.zeros(6, 6, 30),
        )
        factors = torch.full((6, 30), -3.0)
        factors[0] = 3.0
        factors[1] = 2.5
        engine.vm = SimpleNamespace(execute=lambda formula, features: factors)
        engine.best_formula = [1]
        engine.best_score = 1.0
        engine.training_history = {}

        with tempfile.TemporaryDirectory() as directory:
            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                strategy = Path("best_meme_strategy.json")
                strategy.write_text(json.dumps({"formula": [0]}))
                with patch.object(ModelConfig, "TRAIN_STEPS", 0):
                    engine.train()

                self.assertEqual(json.loads(strategy.read_text()), {"formula": [0]})
                self.assertEqual(
                    json.loads(Path("candidate_meme_strategy.json").read_text())["formula"],
                    [1],
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
