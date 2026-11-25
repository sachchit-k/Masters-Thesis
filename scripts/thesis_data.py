"""Data loading for fine-tuning (pandas and numpy only, no torch). Never prints or saves text.

Reads the frozen split ids (data\\splits, made by make_splits.py), checks that the source files
are the ones the splits were made from, and returns frames with columns id, text, label.
DECISIONS: row 1 (splits), 3 (open: all rows kept), 17 (WELFake 0 = real, 1 = fake), 18 (LIAR text only),
19 (LIAR scheme A: 1 = true side, 0 = false side).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job", "state", "party",
             "c_barely_true", "c_false", "c_half_true", "c_mostly_true", "c_pants_fire", "context"]
SCHEME_A = {"pants-fire": 0, "false": 0, "barely-true": 0, "half-true": 1, "mostly-true": 1, "true": 1}
CLASS_NAMES = {
    "liar": ["false_side(0)", "true_side(1)"],
    "welfake": ["real(0)", "fake(1)"],
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(splits_dir: Path) -> dict:
    return json.loads((Path(splits_dir) / "split_manifest.json").read_text())


def check_source_hashes(data_dir: Path, splits_dir: Path, which: str) -> dict:
    """Stop if a source file differs from the one the frozen splits were made from."""
    man = load_manifest(splits_dir)
    names = ["WELFake_Dataset.csv"] if which == "welfake" else [f"liar_dataset/{s}.tsv" for s in ("train", "valid", "test")]
    got = {}
    for n in names:
        got[n] = sha256_file(Path(data_dir) / n)
        if got[n] != man["source_sha256"][n]:
            raise SystemExit(f"STOP: {n} differs from the file the splits were frozen on (sha256 mismatch).")
    return got


def _read_liar_split(data_dir: Path, split: str) -> pd.DataFrame:
    f = Path(data_dir) / "liar_dataset" / f"{split}.tsv"
    return pd.read_csv(f, sep="\t", header=None, names=LIAR_COLS, dtype=str,
                       keep_default_na=False, quoting=csv.QUOTE_NONE, encoding="utf-8")


def load_liar(data_dir, splits_dir, scheme: str = "A"):
    """Statement text only (row 18). Returns ({train, val, test: DataFrame}, info)."""
    if scheme != "A":
        raise SystemExit("Only scheme A is decided (DECISIONS row 19).")
    data_dir, splits_dir = Path(data_dir), Path(splits_dir)
    out, info = {}, {}
    for key, official in (("train", "train"), ("val", "valid"), ("test", "test")):
        raw = _read_liar_split(data_dir, official)
        assert raw["id"].is_unique
        ids = pd.read_csv(splits_dir / f"liar_{official}_ids.csv", dtype=str)["id"]
        d = raw.set_index("id").loc[ids].reset_index()
        d = pd.DataFrame({"id": d["id"], "text": d["statement"], "label": d["label"].map(SCHEME_A)})
        assert d["label"].notna().all(), "unknown LIAR label"
        d["label"] = d["label"].astype(int)
        out[key] = d
        info[key] = {"n": int(len(d)), "label_counts": {int(k): int(v) for k, v in d["label"].value_counts().sort_index().items()}}
    return out, info


def load_welfake(data_dir, splits_dir, text_input: str = "text", drop_empty: bool = True):
    """Rows by position from the frozen ids. text_input: 'text' (default, as in the analysis) or 'title_text'."""
    data_dir, splits_dir = Path(data_dir), Path(splits_dir)
    wf = pd.read_csv(data_dir / "WELFake_Dataset.csv")
    text = wf["text"].fillna("").astype(str)
    if text_input == "title_text":
        full = (wf["title"].fillna("").astype(str) + ". " + text).str.strip(" .")
    elif text_input == "text":
        full = text
    else:
        raise SystemExit("text_input must be 'text' or 'title_text'")
    empty = full.str.strip().eq("").to_numpy()
    out, info = {}, {}
    for key in ("train", "val", "test"):
        rows = pd.read_csv(splits_dir / f"welfake_{key}_ids.csv")["row"].to_numpy()
        d = pd.DataFrame({"id": rows, "text": full.to_numpy()[rows], "label": wf["label"].to_numpy()[rows].astype(int)})
        n0 = len(d)
        skipped = int(empty[rows].sum())
        if drop_empty:
            d = d[~empty[rows]].reset_index(drop=True)
        out[key] = d
        info[key] = {"n_in_split": int(n0), "empty_text_skipped": skipped if drop_empty else 0, "n": int(len(d)),
                     "label_counts": {int(k): int(v) for k, v in d["label"].value_counts().sort_index().items()}}
    return out, info


def load(dataset: str, data_dir, splits_dir, cfg: dict):
    check = check_source_hashes(Path(data_dir), Path(splits_dir), dataset)
    if dataset == "liar":
        frames, info = load_liar(data_dir, splits_dir, cfg.get("label_scheme", "A"))
    elif dataset == "welfake":
        frames, info = load_welfake(data_dir, splits_dir, cfg.get("welfake_input", "text"), cfg.get("drop_empty_text", True))
    else:
        raise SystemExit("dataset must be 'liar' or 'welfake'")
    return frames, info, check


def subset_frames(frames: dict, n_train: int, seed: int) -> dict:
    """Tiny random subset for smoke tests only."""
    rng = np.random.default_rng(seed)
    n_eval = max(20, n_train // 4)
    out = {}
    for k, d in frames.items():
        n = n_train if k == "train" else n_eval
        idx = np.sort(rng.choice(len(d), size=min(n, len(d)), replace=False))
        out[k] = d.iloc[idx].reset_index(drop=True)
    return out
