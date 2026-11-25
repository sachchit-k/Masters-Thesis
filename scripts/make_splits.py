#!/usr/bin/env python
"""Step 4: freeze the split ids (DECISIONS rows 1, 2, 3, 19).

Run from repo\\ inside the project .venv:
    python scripts\\make_splits.py

What it does (counts only, no dataset text is printed or saved):
  LIAR     official train/valid/test split, ids only.
  WELFake  80/10/10, stratified by label, near-duplicate clusters kept together
           (same cluster rule as the data analysis: est. Jaccard >= 0.8, 5-word shingles,
           texts >= 20 words). ALL rows are kept (DECISIONS row 3 is still OPEN); rows with
           empty text get their own group. Seed 42 (DECISIONS row 2).
Outputs in <root>\\data\\splits\\ (never committed):
    liar_{train,valid,test}_ids.csv         column: id
    welfake_{train,val,test}_ids.csv        columns: row (0-based position in the CSV), orig_index
    welfake_clusters.csv (+ .meta.json)     per row: exact_group, near_cluster, group, empty_text, label
    split_manifest.json                     seed, parameters, file hashes, counts per split
A copy of split_manifest.json (no ids, no text) goes to repo\\configs\\ so it can be committed.
Frozen means frozen: if split_manifest.json already exists the script recomputes the splits and
only checks that they are identical. Use --force only if you really intend to re-freeze.
"""
from __future__ import annotations

import argparse, hashlib, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_data_analysis as s4  # reuses loaders and the duplicate detection (no analysis is run on import)
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
ROOT, DATA = s4.ROOT, s4.DATA
SPLITS = DATA / "splits"
REPO = Path(__file__).resolve().parents[1]
CLUSTER_PARAMS = {"shingle_words": 5, "min_words": 20, "est_jaccard": 0.8, "minhash_perms": 128, "bands": 16, "minhash_rng_seed": 0}
SCHEME_A = s4.SCHEMES["binary_A_halftrue_with_true"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(a) -> str:
    return hashlib.sha256(np.asarray(a).astype(str).astype("U").tobytes()).hexdigest()


def liar_splits():
    lr = s4.load_liar()
    assert lr["id"].is_unique, "LIAR ids are not unique"
    out, info = {}, {}
    for sp in ("train", "valid", "test"):
        d = lr[lr["split"] == sp]
        out[sp] = d["id"].to_numpy()
        six = d["label"].value_counts().to_dict()
        a = d["label"].map(SCHEME_A).value_counts().to_dict()
        info[sp] = {"n": int(len(d)), "six_way_counts": {k: int(v) for k, v in six.items()},
                    "scheme_A_counts_0_false_side_1_true_side": {str(k): int(v) for k, v in a.items()}}
    # twins across the official splits (counts only; LIAR keeps its official split, DECISIONS row 1)
    ex, ne = s4.dup_analysis(lr["statement"].tolist(), 3, 4, "liar statements")
    spl = lr["split"].to_numpy()
    tw = {}
    for kind, g in (("exact", ex), ("near", ne)):
        tr = set(g[(spl == "train") & (g >= 0)])
        va = set(g[(spl == "valid") & (g >= 0)])
        tw[kind] = {"valid_rows_with_twin_in_train": int(sum(1 for x in g[spl == "valid"] if x in tr)),
                    "test_rows_with_twin_in_train": int(sum(1 for x in g[spl == "test"] if x in tr)),
                    "test_rows_with_twin_in_train_or_valid": int(sum(1 for x in g[spl == "test"] if x in tr or x in va))}
    info["twins_across_official_splits"] = tw
    return out, info


def welfake_clusters(wf: pd.DataFrame, src_sha: str):
    cache, meta = SPLITS / "welfake_clusters.csv", SPLITS / "welfake_clusters.meta.json"
    want = {"src_sha256": src_sha, "params": CLUSTER_PARAMS, "rows": int(len(wf))}
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == want:
        s4.log("reusing cached WELFake clusters")
        c = pd.read_csv(cache)
        return c["exact_group"].to_numpy(), c["near_cluster"].to_numpy()
    ex, ne = s4.dup_analysis(wf["text"].tolist(), CLUSTER_PARAMS["shingle_words"], CLUSTER_PARAMS["min_words"], "welfake text")
    empty = (wf["text"].isna() | (wf["text"].fillna("").str.strip() == "")).to_numpy()
    n = len(wf)
    group = np.where(ne >= 0, ne, n + np.arange(n))
    pd.DataFrame({"row": np.arange(n), "label": wf["label"].to_numpy(), "exact_group": ex, "near_cluster": ne,
                  "group": group, "empty_text": empty.astype(int)}).to_csv(cache, index=False)
    meta.write_text(json.dumps(want, indent=1))
    return ex, ne


def welfake_splits(src_sha: str):
    wf = s4.load_welfake()
    n = len(wf)
    ex, ne = welfake_clusters(wf, src_sha)
    empty = (wf["text"].isna() | (wf["text"].fillna("").str.strip() == "")).to_numpy()
    group = np.where(ne >= 0, ne, n + np.arange(n))  # empty texts: own group
    y = wf["label"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = [te for _, te in sgkf.split(np.zeros(n), y, group)]
    test, val = np.sort(folds[0]), np.sort(folds[1])
    train = np.sort(np.concatenate(folds[2:]))
    idx = {"train": train, "val": val, "test": test}
    # checks: disjoint, complete, no cluster and no exact-duplicate group in two splits
    allr = np.concatenate(list(idx.values()))
    assert len(allr) == n and len(np.unique(allr)) == n, "splits are not a partition"
    of = np.empty(n, dtype=object)
    for k, v in idx.items():
        of[v] = k
    for name, g in (("near cluster", group), ("exact group", np.where(ex >= 0, ex, -1 - np.arange(n)))):
        s = pd.DataFrame({"g": g, "sp": of}).groupby("g")["sp"].nunique()
        assert (s <= 1).all(), f"a {name} spans more than one split"
    orig = wf["orig_index"].to_numpy() if "orig_index" in wf.columns else np.arange(n)
    gsize = pd.Series(group).map(pd.Series(group).value_counts()).to_numpy()
    info = {}
    for k, v in idx.items():
        yy = y[v]
        info[k] = {"n": int(len(v)), "share_of_rows": round(len(v) / n, 4), "label0_real": int((yy == 0).sum()),
                   "label1_fake": int((yy == 1).sum()), "share_label1": round(float((yy == 1).mean()), 4),
                   "empty_text_rows": int(empty[v].sum()), "rows_in_duplicate_clusters": int((gsize[v] > 1).sum())}
    info["clusters"] = {"groups": int(len(np.unique(group))), "groups_with_2plus_rows": int((pd.Series(group).value_counts() > 1).sum()),
                        "largest_group": int(pd.Series(group).value_counts().max()), "empty_text_rows_total": int(empty.sum())}
    return idx, orig, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite frozen splits (only if you intend to re-freeze)")
    args = ap.parse_args()
    SPLITS.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLITS / "split_manifest.json"

    src = {"WELFake_Dataset.csv": sha256_file(DATA / "WELFake_Dataset.csv")}
    for sp in ("train", "valid", "test"):
        src[f"liar_dataset/{sp}.tsv"] = sha256_file(DATA / "liar_dataset" / f"{sp}.tsv")

    s4.log("LIAR")
    liar_ids, liar_info = liar_splits()
    s4.log("WELFake")
    wf_idx, orig, wf_info = welfake_splits(src["WELFake_Dataset.csv"])

    hashes = {f"liar_{k}": sha256_ids(v) for k, v in liar_ids.items()}
    hashes.update({f"welfake_{k}": sha256_ids(v) for k, v in wf_idx.items()})

    if manifest_path.exists() and not args.force:
        old = json.loads(manifest_path.read_text())
        if old.get("id_hashes") == hashes and old.get("source_sha256") == src:
            print("FROZEN SPLITS VERIFIED: recomputed splits are identical to the frozen ones.")
            return
        print("ERROR: recomputed splits DIFFER from the frozen ones (data file, seed or code changed). Nothing written. "
              "Use --force only if you intend to re-freeze (and then retrain).")
        sys.exit(1)

    for k, v in liar_ids.items():
        pd.DataFrame({"id": v}).to_csv(SPLITS / f"liar_{k}_ids.csv", index=False)
    for k, v in wf_idx.items():
        pd.DataFrame({"row": v, "orig_index": orig[v]}).to_csv(SPLITS / f"welfake_{k}_ids.csv", index=False)

    manifest = {
        "seed": SEED, "decisions": "rows 1, 2, 3 (OPEN: all rows kept), 19",
        "welfake_method": "StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42): fold 0 = test, fold 1 = val, rest = train; groups = near-duplicate clusters",
        "cluster_params": CLUSTER_PARAMS, "source_sha256": src, "id_hashes": hashes,
        "liar": liar_info, "welfake": wf_info,
        "note": "ids only; LIAR scheme A (DECISIONS row 19) is applied at load time; WELFake label 0 = real, 1 = fake (row 17); empty-text rows are kept in the ids (row 3 open)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=1))
    (REPO / "configs").mkdir(exist_ok=True)
    shutil.copy(manifest_path, REPO / "configs" / "split_manifest.json")
    print(json.dumps({"liar": {k: liar_info[k]["n"] for k in ("train", "valid", "test")},
                      "welfake": {k: wf_info[k]["n"] for k in ("train", "val", "test")},
                      "welfake_share_label1": {k: wf_info[k]["share_label1"] for k in ("train", "val", "test")},
                      "liar_twins": liar_info["twins_across_official_splits"]}, indent=1))
    print(f"Frozen. Manifest: {manifest_path}  (copy in repo\\configs\\split_manifest.json, commit that one)")


if __name__ == "__main__":
    main()
