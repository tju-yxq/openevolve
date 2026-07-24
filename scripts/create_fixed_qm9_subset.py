#!/usr/bin/env python
"""Create a deterministic fixed subset of the existing QM9 training split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_bytes(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value.astype(np.int64, copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--fraction", type=float, default=0.25)
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    data_path = Path(args.data_path).resolve()
    split_path = data_path / "splits.npz"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    with np.load(split_path) as split:
        train_global = np.asarray(split["idx_train"], dtype=np.int64)
        valid_global = np.asarray(split["idx_valid"], dtype=np.int64)
        test_global = np.asarray(split["idx_test"], dtype=np.int64)

    if len(np.unique(train_global)) != len(train_global):
        raise ValueError("the source training split contains duplicate indices")
    if np.intersect1d(train_global, valid_global).size:
        raise ValueError("source train and validation splits overlap")
    if np.intersect1d(train_global, test_global).size:
        raise ValueError("source train and test splits overlap")

    # QM9 processing writes training molecules in ascending original-dataset
    # order. These local positions therefore index train_one_hot.pt directly.
    train_global_in_processed_order = np.sort(train_global)
    subset_size = int(len(train_global) * args.fraction)
    rng = np.random.default_rng(args.seed)
    train_local_indices = np.sort(
        rng.choice(len(train_global), size=subset_size, replace=False)
    ).astype(np.int64)
    train_global_indices = train_global_in_processed_order[train_local_indices]

    if len(train_local_indices) != subset_size:
        raise AssertionError("unexpected subset size")
    if len(np.unique(train_local_indices)) != subset_size:
        raise AssertionError("the generated subset contains duplicates")
    if np.intersect1d(train_global_indices, valid_global).size:
        raise AssertionError("generated subset overlaps validation")
    if np.intersect1d(train_global_indices, test_global).size:
        raise AssertionError("generated subset overlaps test")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        train_local_indices=train_local_indices,
        train_global_indices=train_global_indices,
        source_train_global_indices=train_global,
        seed=np.asarray(args.seed, dtype=np.int64),
        fraction=np.asarray(args.fraction, dtype=np.float64),
    )

    manifest = {
        "artifact_version": 1,
        "dataset": "QM9",
        "source_split": str(split_path),
        "source_split_sha256": sha256_file(split_path),
        "source_train_size": int(len(train_global)),
        "source_validation_size": int(len(valid_global)),
        "source_test_size": int(len(test_global)),
        "subset_fraction": float(args.fraction),
        "subset_seed": int(args.seed),
        "subset_sampling": "numpy.default_rng(seed).choice without replacement",
        "subset_size": int(subset_size),
        "local_index_semantics": "Indices into the processed QM9 train split.",
        "train_local_indices_sha256": sha256_bytes(train_local_indices),
        "train_global_indices_sha256": sha256_bytes(train_global_indices),
        "unique_local_indices": int(len(np.unique(train_local_indices))),
        "validation_overlap": int(np.intersect1d(train_global_indices, valid_global).size),
        "test_overlap": int(np.intersect1d(train_global_indices, test_global).size),
        "npz_file": output.name,
        "npz_sha256": sha256_file(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
