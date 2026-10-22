"""Reference arithmetic for the SHL 2026 paper's selected probability fusion.

New companion implementation; not the original competition pipeline.
Inputs are aligned, already normalized per-model class probabilities.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


CLASS_NAMES = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]
WEIGHTS = {"NuTime": 0.54, "Mantis": 0.29, "TiRex": 0.17}
ROW_SUM_TOLERANCE = 1e-6


def _validate_matrix(name: str, matrix: object) -> list[list[float]]:
    if not isinstance(matrix, list) or not matrix:
        raise ValueError(f"{name}: expected a nonempty N x 8 JSON array")
    validated = []
    for index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != len(CLASS_NAMES):
            raise ValueError(f"{name} row {index}: expected exactly 8 probabilities")
        numbers = []
        for column, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} row {index} column {column}: expected a number, not a boolean or string")
            try:
                number = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{name} row {index} column {column}: probability is not finite") from exc
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} row {index} column {column}: probability must be finite and in [0, 1]")
            numbers.append(number)
        if not math.isclose(math.fsum(numbers), 1.0, rel_tol=0.0, abs_tol=ROW_SUM_TOLERANCE):
            raise ValueError(f"{name} row {index}: probabilities must sum to 1 within {ROW_SUM_TOLERANCE:g}")
        validated.append(numbers)
    return validated


def fuse_probabilities(inputs: object) -> dict:
    """Validate aligned N x 8 matrices, fuse fixed weights, return labels.

    Rows must represent the same samples in the same order in all three inputs.
    No normalization, calibration, weight selection or model inference is done.
    Exact argmax ties select the earliest class in CLASS_NAMES.
    """
    if not isinstance(inputs, dict) or set(inputs) != set(WEIGHTS):
        raise ValueError("Expected exactly three model keys: NuTime, Mantis, TiRex")
    matrices = {name: _validate_matrix(name, inputs[name]) for name in WEIGHTS}
    count = len(matrices["NuTime"])
    if any(len(matrix) != count for matrix in matrices.values()):
        raise ValueError("All three models must have the same number of rows")
    probabilities = [
        [math.fsum(WEIGHTS[name] * matrices[name][row][column] for name in WEIGHTS)
         for column in range(len(CLASS_NAMES))]
        for row in range(count)
    ]
    indices = [max(range(len(CLASS_NAMES)), key=row.__getitem__) for row in probabilities]
    return {
        "method": "published_fixed_probability_fusion_reference",
        "weights": dict(WEIGHTS),
        "class_names": list(CLASS_NAMES),
        "probabilities": probabilities,
        "label_indices": indices,
        "labels": [CLASS_NAMES[index] for index in indices],
    }


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON with NuTime, Mantis and TiRex probability arrays")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON; its parent directory must exist")
    parser.add_argument("--force", action="store_true", help="Allow overwriting the output file")
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output paths must differ, including with --force")
        inputs = json.loads(args.input.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_keys)
        result = fuse_probabilities(inputs)
        content = json.dumps(result, indent=2, allow_nan=False) + "\n"
        try:
            with args.output.open("w" if args.force else "x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
        except FileExistsError as exc:
            raise ValueError("Output already exists; choose another path or pass --force") from exc
        print(f"Fused {len(result['labels'])} rows into {args.output}. No performance metric was computed.")
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
