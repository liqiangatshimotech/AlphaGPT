import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import math

from .config import ModelConfig
from .data_loader import CryptoDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MemeBacktest
from .factors import FeatureEngineer
from .vocab import FORMULA_VOCAB, FORMULA_VOCAB_VERSION, load_formula

CANDIDATE_STRATEGY_PATH = "candidate_meme_strategy.json"

class AlphaEngine:
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        """
        Initialize AlphaGPT training engine.
        
        Args:
            use_lord_regularization: Enable Low-Rank Decay (LoRD) regularization
            lord_decay_rate: Strength of LoRD regularization
            lord_num_iterations: Number of Newton-Schulz iterations per step
        """
        self.loader = CryptoDataLoader()
        self.loader.load_data()
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # Standard optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Low-Rank Decay regularizer
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = MemeBacktest()
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }

    def train(self):
        print("🚀 Starting Meme Alpha Mining with LoRD Regularization..." if self.use_lord else "🚀 Starting Meme Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")

        # The final window is never used for policy-gradient rewards or best
        # candidate selection. The two-candle gap keeps training labels from
        # reaching into the first holdout candle.
        training_window, holdout_window = self.bt.temporal_windows(
            self.loader.target_ret.shape[1]
        )
        training_data = {
            name: values[:, training_window]
            for name, values in self.loader.raw_data_cache.items()
        }
        training_target = self.loader.target_ret[:, training_window]
        # Recompute features using training data alone. The loader's full
        # tensor includes the reserved holdout in its normalization statistics.
        training_features = FeatureEngineer.compute_features(training_data)
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                res = self.vm.execute(formula, training_features)
                
                if res is None:
                    rewards[i] = -5.0
                    continue

                quality_ok, _ = self.bt.check_candidate_quality(
                    res, training_data
                )
                if not quality_ok:
                    rewards[i] = -5.0
                    continue

                score, ret_val = self.bt.evaluate(
                    res, training_data, training_target,
                )
                score_value = score.item()
                if not (math.isfinite(score_value) and math.isfinite(ret_val)
                        and score_value > 0 and ret_val > 0):
                    rewards[i] = -5.0
                    continue
                rewards[i] = score

                if score_value > self.best_score:
                    self.best_score = score_value
                    self.best_formula = formula
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula}")
            
            # Normalize rewards
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            
            loss = loss.mean()
            
            # Gradient step
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            # Apply Low-Rank Decay regularization
            if self.use_lord:
                self.lord_opt.step()
            
            # Logging
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)

        # Test only the selected candidate on the untouched final time window.
        # Never promote a training result into the live strategy automatically.
        if self.best_formula is None:
            raise RuntimeError("No formula passed score-dispersion and training-return checks")
        load_formula(self.best_formula)
        best_factors = self.vm.execute(self.best_formula, self.loader.feat_tensor)
        if best_factors is None:
            raise RuntimeError("Selected formula cannot be evaluated")
        quality_ok, reason = self.bt.check_candidate_quality(
            best_factors, self.loader.raw_data_cache
        )
        if not quality_ok:
            raise RuntimeError(f"Selected formula failed latest-score validation: {reason}")
        holdout_score, holdout_return = self.bt.evaluate(
            best_factors, self.loader.raw_data_cache, self.loader.target_ret,
            period=holdout_window,
        )
        if not (math.isfinite(holdout_score.item())
                and math.isfinite(holdout_return)
                and holdout_score.item() > 0 and holdout_return > 0):
            raise RuntimeError(
                "Selected formula failed the final time holdout "
                f"(score={holdout_score.item():.4f}, mean_return={holdout_return:.4f})"
            )
        # Save a reviewable candidate; live promotion is a separate action.
        with open(CANDIDATE_STRATEGY_PATH, "w") as f:
            json.dump({
                "formula": self.best_formula,
                "vocab_version": FORMULA_VOCAB_VERSION,
                "token_names": list(FORMULA_VOCAB.token_names),
            }, f)
        
        # Save training history
        import json as js
        with open("training_history.json", "w") as f:
            js.dump(self.training_history, f)
        
        print(f"\n✓ Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        print(f"  Candidate saved: {CANDIDATE_STRATEGY_PATH}")


if __name__ == "__main__":
    eng = AlphaEngine(use_lord_regularization=True)
    eng.train()
