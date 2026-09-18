"""Input validation and the exact preprocessing used at training time
(scripts/train_model.py, mirroring assets/earthquake_damage.ipynb's final
LightGBM cell): geo columns stay raw int, the 8 low-cardinality string
columns are cast to pandas `category` dtype. No scaling, no encoding, no
feature engineering -- this model was trained on raw columns only.
"""

import pandas as pd

from src.feature_metadata import load_feature_metadata


class ValidationError(ValueError):
    """Raised when a building record fails schema/range/category validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_input(values: dict) -> list[str]:
    """Validate a raw {feature_name: value} dict against the training-data
    schema. Returns a list of human-readable error strings (empty = valid)."""
    meta = load_feature_metadata()
    errors: list[str] = []

    required = meta["feature_order"]
    missing = [c for c in required if c not in values or values[c] is None or values[c] == ""]
    if missing:
        errors.append(
            "Missing required field(s): " + ", ".join(meta["features"][c]["display_name"] for c in missing)
        )

    for col in meta["geo_cols"]:
        if col in values and values[col] not in (None, ""):
            fmeta = meta["features"][col]
            try:
                v = int(values[col])
            except (TypeError, ValueError):
                errors.append(f"{fmeta['display_name']} must be an integer.")
                continue
            if not (fmeta["min"] <= v <= fmeta["max"]):
                errors.append(
                    f"{fmeta['display_name']} must be between {fmeta['min']} and {fmeta['max']} "
                    f"(the range seen in training data)."
                )

    for col in meta["numeric_cols"]:
        if col in values and values[col] not in (None, ""):
            fmeta = meta["features"][col]
            try:
                v = float(values[col])
            except (TypeError, ValueError):
                errors.append(f"{fmeta['display_name']} must be a number.")
                continue
            if not (fmeta["min"] <= v <= fmeta["max"]):
                errors.append(
                    f"{fmeta['display_name']} must be between {fmeta['min']} and {fmeta['max']} "
                    f"(the range seen in training data)."
                )

    for col in meta["categorical_cols"]:
        if col in values and values[col] not in (None, ""):
            fmeta = meta["features"][col]
            valid_codes = {v["code"] for v in fmeta["values"]}
            if values[col] not in valid_codes:
                errors.append(
                    f"{fmeta['display_name']} value {values[col]!r} is not one of the categories "
                    f"seen in training: {sorted(valid_codes)}."
                )

    binary_cols = meta["superstructure_cols"] + [meta["secondary_use_umbrella_col"]] + meta["secondary_use_sub_cols"]
    for col in binary_cols:
        if col in values and values[col] not in (None, ""):
            try:
                v = int(values[col])
            except (TypeError, ValueError):
                v = None
            if v not in (0, 1):
                errors.append(f"{meta['features'][col]['display_name']} must be 0 or 1.")

    return errors


def build_input_dataframe(values: dict) -> pd.DataFrame:
    """Build the exact one-row DataFrame schema the model expects: the 38
    training columns, in the exact training column order, with the 8
    categorical columns cast to pandas `category` dtype (matching
    scripts/train_model.py's cast_categoricals). Raises ValidationError if
    `values` fails schema validation."""
    errors = validate_input(values)
    if errors:
        raise ValidationError(errors)

    meta = load_feature_metadata()
    row = {}
    for col in meta["feature_order"]:
        v = values[col]
        if col in meta["categorical_cols"]:
            row[col] = v
        elif col in meta["geo_cols"] or col in meta["numeric_cols"]:
            row[col] = int(v)
        else:  # binary flags
            row[col] = int(v)

    df = pd.DataFrame([row], columns=meta["feature_order"])
    for col in meta["categorical_cols"]:
        df[col] = df[col].astype("category")
    return df
