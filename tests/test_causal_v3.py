"""Causality checks for the opt-in v3 research path."""

import unittest

import torch

from model_core.causal_v3 import CausalV3FeatureEngineer, CausalV3StackVM


class CausalV3Tests(unittest.TestCase):
    def setUp(self):
        length = 48
        t = torch.arange(length, dtype=torch.float32).expand(2, -1)
        close = 1.0 + t * 0.01 + torch.sin(t * 0.47) * 0.03
        close = close + torch.tensor([[0.0], [0.4]])
        close[:, 36] += 1.0  # A future jump must not change any past factor.
        open_ = close * 0.995
        self.raw = {
            "open": open_,
            "high": close * 1.01,
            "low": open_ * 0.99,
            "close": close,
            "volume": 100.0 + 10.0 * torch.sin(t * 0.83) + t,
            "liquidity": 700_000.0 + t * 100.0,
            "fdv": 2_000_000.0 + t * 1000.0,
        }
        self.observed = torch.ones((2, length), dtype=torch.bool)
        self.observed[0, [0, 8, 9, 21, 22, 37]] = False
        self.observed[1, [2, 3, 4, 20, 31, 32]] = False
        self.engineer = CausalV3FeatureEngineer()
        self.vm = CausalV3StackVM()

    def test_feature_and_operator_outputs_are_prefix_invariant(self):
        full = self.engineer.compute_features(self.raw, self.observed)
        formulas = (
            [0],       # RET
            [0, 14],   # JUMP
            [0, 15],   # DECAY
            [0, 16],   # DELAY1
            [0, 17],   # MAX3
            [2, 0, 9], # guarded DIV
        )
        full_results = [self.vm.execute(formula, full, self.observed) for formula in formulas]
        for end in (1, 2, 3, 5, 12, 26, 35, 48):
            with self.subTest(end=end):
                raw = {name: values[:, :end] for name, values in self.raw.items()}
                observed = self.observed[:, :end]
                prefix = self.engineer.compute_features(raw, observed)
                torch.testing.assert_close(prefix, full[:, :, :end], rtol=1e-5, atol=1e-6)
                for formula, expected in zip(formulas, full_results):
                    result = self.vm.execute(formula, prefix, observed)
                    self.assertIsNotNone(result)
                    torch.testing.assert_close(result, expected[:, :end], rtol=1e-5, atol=1e-6)

    def test_unobserved_raw_values_are_ignored(self):
        expected = self.engineer.compute_features(self.raw, self.observed)
        poisoned = {
            name: torch.where(self.observed, values, 1e30)
            for name, values in self.raw.items()
        }
        actual = self.engineer.compute_features(poisoned, self.observed)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.equal(actual.masked_select(~self.observed[:, None, :]), torch.zeros_like(actual.masked_select(~self.observed[:, None, :]))))

    def test_delay_skips_missing_candles_and_division_guards_zero(self):
        observed = torch.tensor([[True, False, False, True, False, True]])
        features = torch.zeros((1, 6, 6))
        features[0, 0] = torch.tensor([2.0, 999.0, 999.0, 3.0, 999.0, 4.0])
        features[0, 1] = 1.0
        delay = self.vm.execute([0, 16], features, observed)
        torch.testing.assert_close(delay, torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0, 3.0]]))

        denominator = torch.zeros_like(features)
        denominator[:, 2] = features[:, 1]
        divided = self.vm.execute([2, 0, 9], denominator, observed)
        self.assertIsNotNone(divided)
        self.assertTrue(torch.equal(divided, torch.zeros_like(divided)))

    def test_historical_liquidity_feature_is_disabled(self):
        features = self.engineer.compute_features(self.raw, self.observed)
        self.assertTrue(torch.equal(features[:, 1], torch.zeros_like(features[:, 1])))
        self.assertIsNone(self.vm.execute([1], features, self.observed))
        self.assertIsNone(self.vm.execute(
            [1, 4, 2, 14, 17, 12, 8, 14, 9, 1, 9, 11],
            features, self.observed,
        ))

    def test_malformed_observed_candle_fails_closed(self):
        raw = {name: value.clone() for name, value in self.raw.items()}
        raw["open"][0, 1] = 0.0
        with self.assertRaisesRegex(ValueError, "observed open must be positive"):
            self.engineer.compute_features(raw, self.observed)

    def test_extreme_finite_price_jump_keeps_return_finite(self):
        close = torch.tensor([[1e-20, 1e29, 1e29, 1e29, 1e29, 1e29]])
        raw = {name: close.clone() for name in ("open", "high", "low", "close")}
        raw["volume"] = torch.ones_like(close)
        observed = torch.ones_like(close, dtype=torch.bool)

        features = self.engineer.compute_features(raw, observed)
        result = self.vm.execute([0], features, observed)

        self.assertTrue(torch.isfinite(features).all().item())
        self.assertIsNotNone(result)
        self.assertTrue(torch.isfinite(result).all().item())

    def test_direct_feature_formula_rejects_nonfinite_observation(self):
        features = torch.zeros((1, 6, 6))
        features[0, 0, 2] = float("nan")
        observed = torch.ones((1, 6), dtype=torch.bool)
        self.assertIsNone(self.vm.execute([0], features, observed))


if __name__ == "__main__":
    unittest.main()
