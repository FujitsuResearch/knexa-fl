#!/usr/bin/env python3
"""Deterministic split serializer for HumanEval + MBPP.

Reads local dataset files (JSONL) and the six‑client roster, then writes
per‑client train/val splits and a global test set to `artifacts/data/splits/`.

This does not download data; you must provide local paths.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                # Skip malformed lines
                continue
    return items


def _get_item_id(obj: Dict, prefix: str, idx: int) -> str:
    # Prefer common fields; fallback to generated ID
    for k in ("task_id", "id", "problem_id"):
        v = obj.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, int):
            return f"{prefix}_{v}"
    return f"{prefix}_{idx}"


def _load_datasets(human_eval_path: Path, mbpp_path: Path) -> Tuple[List[Dict], List[Dict]]:
    he_items = _read_jsonl(human_eval_path)
    mb_items = _read_jsonl(mbpp_path)
    return he_items, mb_items


def _combine_with_ids(he_items: List[Dict], mb_items: List[Dict]) -> List[Dict]:
    combined = []
    for i, x in enumerate(he_items):
        x = dict(x)
        x["__id"] = _get_item_id(x, "he", i)
        x["__src"] = "HumanEval"
        combined.append(x)
    for i, x in enumerate(mb_items):
        x = dict(x)
        x["__id"] = _get_item_id(x, "mbpp", i)
        x["__src"] = "MBPP"
        combined.append(x)
    # Deduplicate by __id if any collisions
    seen = set()
    uniq = []
    for x in combined:
        if x["__id"] in seen:
            continue
        seen.add(x["__id"])
        uniq.append(x)
    return uniq


def _load_roster(roster_path: Path) -> Dict:
    with roster_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _alloc_splits(
    combined_ids: List[str],
    roster: Dict,
    seed: int,
    global_test_size: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """Allocate per‑client train/val and a global test set deterministically.

    - Samples without replacement.
    - Validates counts and fails if pool is insufficient.
    """
    rng = np.random.default_rng(seed)
    pool = combined_ids.copy()
    rng.shuffle(pool)

    # Compute required totals
    req_train = sum(int(c.get("train_samples", 0)) for c in roster.get("clients", []))
    req_val = sum(int(c.get("val_samples", 0)) for c in roster.get("clients", []))
    req_total = req_train + req_val + int(global_test_size)
    if len(pool) < req_total:
        raise RuntimeError(
            f"Insufficient dataset size: pool={len(pool)} required={req_total} "
            f"(train={req_train}, val={req_val}, test={global_test_size})"
        )

    # Allocate train per client
    train_splits: Dict[str, List[str]] = {}
    val_splits: Dict[str, List[str]] = {}

    cursor = 0
    for c in roster.get("clients", []):
        cid = c["id"]
        n = int(c.get("train_samples", 0))
        train_splits[cid] = pool[cursor : cursor + n]
        cursor += n
    # Allocate val per client
    for c in roster.get("clients", []):
        cid = c["id"]
        n = int(c.get("val_samples", 0))
        val_splits[cid] = pool[cursor : cursor + n]
        cursor += n

    # Allocate global test set
    test_ids = pool[cursor : cursor + int(global_test_size)]
    cursor += int(global_test_size)

    return train_splits, val_splits, test_ids


def main():
    ap = argparse.ArgumentParser(description="Create deterministic splits for HumanEval+MBPP")
    ap.add_argument("--human-eval", required=True, help="Path to HumanEval JSONL file")
    ap.add_argument("--mbpp", required=True, help="Path to MBPP JSONL file")
    ap.add_argument("--roster", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "roster" / "client_roster.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--global-test-size", type=int, default=116)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "data" / "splits"))
    args = ap.parse_args()

    he_path = Path(args.human_eval)
    mb_path = Path(args.mbpp)
    roster_path = Path(args.roster)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    he_items, mb_items = _load_datasets(he_path, mb_path)
    combined = _combine_with_ids(he_items, mb_items)
    combined_ids = [x["__id"] for x in combined]

    roster = _load_roster(roster_path)

    train_splits, val_splits, test_ids = _alloc_splits(
        combined_ids, roster, seed=args.seed, global_test_size=args.global_test_size
    )

    # Write outputs
    (out_dir / "combined_ids.json").write_text(json.dumps(combined_ids, indent=2) + "\n", encoding="utf-8")
    (out_dir / "global_test.json").write_text(json.dumps(test_ids, indent=2) + "\n", encoding="utf-8")
    for cid, ids in train_splits.items():
        (out_dir / f"client_{cid}_train.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")
    for cid, ids in val_splits.items():
        (out_dir / f"client_{cid}_val.json").write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    print("Wrote splits to:", str(out_dir))


if __name__ == "__main__":
    main()

