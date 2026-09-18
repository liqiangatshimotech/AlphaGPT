from dataclasses import dataclass

from .ops import OPS_CONFIG


FEATURE_NAMES = (
    "RET",
    "LIQ_SCORE",
    "PRESSURE",
    "FOMO",
    "DEV",
    "LOG_VOL",
)

# Version the persisted format. Unversioned formulas were evaluated by a VM
# with six features too; preserve those numeric tokens without shifting them.
FORMULA_VOCAB_VERSION = 2


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)


def load_formula(data):
    """Validate a saved formula without guessing or remapping its vocabulary."""
    if isinstance(data, list):
        formula = data
    elif isinstance(data, dict):
        if "vocab_version" in data:
            version = data["vocab_version"]
            if type(version) is not int or version != FORMULA_VOCAB_VERSION:
                raise ValueError(f"Unsupported strategy vocabulary version: {version!r}")
            if data.get("token_names") != list(FORMULA_VOCAB.token_names):
                raise ValueError("Strategy token vocabulary does not match this runtime")
        formula = data.get("formula")
    else:
        raise ValueError("Strategy must contain a formula list or object")
    if not isinstance(formula, list) or not formula:
        raise ValueError("Strategy formula is missing or empty")
    depth = 0
    for token in formula:
        if type(token) is not int or not 0 <= token < FORMULA_VOCAB.size:
            raise ValueError(f"Invalid formula token: {token!r}")
        if token < FORMULA_VOCAB.operator_offset:
            depth += 1
        else:
            arity = OPS_CONFIG[token - FORMULA_VOCAB.operator_offset][2]
            if depth < arity:
                raise ValueError("Formula has insufficient operands")
            depth += 1 - arity
    if depth != 1:
        raise ValueError("Formula must produce exactly one result")
    return list(formula)
