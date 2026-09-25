"""Versioned offline candidate artifacts for the causal v3 research path.

A v3 candidate uses the same numeric token ids as v2, but the features and
operators have different semantics (observed-only rolling statistics, guarded
division, disabled LIQ_SCORE). The artifact therefore records every version
that changes how a formula becomes a trade, and ``load_v3_candidate`` refuses
anything else. ``model_core.vocab.load_formula`` (used by the live runner)
accepts only ``vocab_version == 2`` and so rejects these files as well.

Artifacts are written only to research output directories. Writing to a live
strategy file name is refused so a failed or unreviewed search cannot replace
the running strategy.
"""

import hashlib
import json
import math
import os
from pathlib import Path

from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB


ARTIFACT_TYPE = "alphagpt_causal_v3_candidate"
ARTIFACT_SCHEMA_VERSION = 1
FEATURE_VERSION = "causal_v3.features.1"
VM_VERSION = "causal_v3.vm.1"
SIGNAL_VERSION = "sigmoid_threshold.1"
EXIT_POLICY_VERSION = "hard_rules.1"
# v3 keeps the v2 token order but not its meaning; a new number keeps the live
# loader (which accepts only 2) from interpreting a v3 formula.
V3_VOCAB_VERSION = 3
DISABLED_TOKENS = (1,)  # LIQ_SCORE: no trusted as-of historical source.

PROTECTED_STRATEGY_NAMES = frozenset({
    "best_meme_strategy.json",
    "candidate_meme_strategy.json",
})


def runtime_versions():
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "vocab_version": V3_VOCAB_VERSION,
        "feature_version": FEATURE_VERSION,
        "vm_version": VM_VERSION,
        "signal_version": SIGNAL_VERSION,
        "exit_policy_version": EXIT_POLICY_VERSION,
        "token_names": list(FORMULA_VOCAB.token_names),
        "disabled_tokens": list(DISABLED_TOKENS),
    }


def validate_v3_formula(formula):
    """Check a postfix formula is well formed and uses no disabled token."""
    if not isinstance(formula, list) or not formula:
        raise ValueError("formula must be a non-empty list")
    depth = 0
    for token in formula:
        if type(token) is not int or not 0 <= token < FORMULA_VOCAB.size:
            raise ValueError(f"invalid formula token: {token!r}")
        if token in DISABLED_TOKENS:
            raise ValueError(f"formula uses disabled token {token}")
        if token < FORMULA_VOCAB.operator_offset:
            depth += 1
            continue
        arity = OPS_CONFIG[token - FORMULA_VOCAB.operator_offset][2]
        if depth < arity:
            raise ValueError("formula has insufficient operands")
        depth += 1 - arity
    if depth != 1:
        raise ValueError("formula must produce exactly one result")
    return list(formula)


def _check_output_path(path):
    path = Path(path)
    if path.name in PROTECTED_STRATEGY_NAMES:
        raise ValueError(f"refusing to write research output to live strategy name {path.name}")
    return path


def formula_hash(formula):
    return hashlib.sha256(json.dumps(formula).encode()).hexdigest()[:16]


def build_candidate(formula, *, signal, exit_policy, training, selection):
    validate_v3_formula(formula)
    threshold = signal.get("buy_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("buy_threshold must be in (0, 1)")
    return {
        **runtime_versions(),
        "formula": list(formula),
        "formula_hash": formula_hash(formula),
        "signal": dict(signal),
        "exit_policy": dict(exit_policy),
        "training": training,
        "selection": selection,
        "live_promotion": "not_promoted",
    }


def _json_default(value):
    """Convert NumPy scalars/arrays and paths to native JSON types.

    Anything else is an error: stringifying (the old ``default=str``) turned
    ``np.bool_(False)`` into the truthy string ``"False"``.
    """
    import numpy as np

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__} to JSON: {value!r}")


def write_json_atomic(path, payload):
    path = _check_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def load_v3_candidate(data):
    """Return a validated v3 candidate dict or raise ValueError."""
    if isinstance(data, (str, Path)):
        with open(data) as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("v3 candidate must be a JSON object (v2 formula lists are not accepted)")
    if data.get("revocation") is not None:
        raise ValueError(f"v3 candidate was revoked: {data['revocation'].get('reason')}")
    expected = runtime_versions()
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"incompatible v3 candidate: {key}={data.get(key)!r}, expected {value!r}")
    validate_v3_formula(data.get("formula"))
    signal = data.get("signal")
    if not isinstance(signal, dict) or not 0 < float(signal.get("buy_threshold", -1)) < 1:
        raise ValueError("v3 candidate has no valid buy threshold")
    if signal.get("ai_exit_threshold") is not None:
        raise ValueError("v3 candidates use hard-rule exits only; AI exit thresholds are unsupported")
    if not isinstance(data.get("exit_policy"), dict):
        raise ValueError("v3 candidate has no exit policy")
    return data
