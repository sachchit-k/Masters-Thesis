#!/usr/bin/env python
"""Step 5: fine-tune RoBERTa-base on one dataset (LIAR scheme A or WELFake). Counts and metrics only: no text is printed or saved.

Real run (Kaggle):  python scripts/train.py --config configs/train_liar.yaml --data-dir <data> --out-root <out>
Smoke test (laptop): python scripts\\train.py --config configs\\train_liar.yaml --subset 200 --epochs 1 --max-length 128 --run-name smoke_liar
Outputs:  <out-root>/models/<run>/best/        best checkpoint (by validation macro-F1)
          <out-root>/outputs/training/<run>/   metrics.json, predictions.npz (ids and logits of val and test)
Test split: evaluated ONCE, after the best checkpoint is chosen on validation.
DECISIONS: 2 (seed 42), 4 (metrics), 16 (512, fp16, Kaggle), 18 (LIAR text only, macro-F1), 19 (scheme A).
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_data as td

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("THESIS_ROOT", REPO.parent))
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(tok, texts, max_length: int, chunk: int = 2000):
    """Token ids without padding. Truncation keeps the beginning of the text (DECISIONS row 16)."""
    tok.truncation_side = "right"
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(tok(list(texts[i:i + chunk]), truncation=True, max_length=max_length, padding=False)["input_ids"])
    return out


def collate(seqs, pad_id: int, device):
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        att[i, :len(s)] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def predict_logits(model, seqs, bs: int, pad_id: int, device, use_amp: bool) -> np.ndarray:
    """Logits in the original order (batches are built from length-sorted texts for speed)."""
    model.eval()
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        ids, att = collate([seqs[j] for j in idx], pad_id, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            lg = model(input_ids=ids, attention_mask=att).logits
        out[idx] = lg.float().cpu().numpy()
    return out


def score(y, pred) -> dict:
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "accuracy": float((y == pred).mean())}


def full_metrics(y, logits, train_majority: int, n_boot: int, seed: int, class_names) -> dict:
    pred = logits.argmax(1)
    m = score(y, pred)
    maj = np.full_like(y, train_majority)
    m["majority_baseline"] = score(y, maj)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    m["per_class"] = {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in (0, 1)}
    m["confusion_matrix_rows_true_cols_pred"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    rng = np.random.default_rng(seed)
    n = len(y)
    f1s, accs = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        sc = score(y[ix], pred[ix])
        f1s.append(sc["macro_f1"]); accs.append(sc["accuracy"])
    m["bootstrap_95ci"] = {"macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
                           "accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))], "resamples": n_boot}
    m["n"] = int(n)
    return m


def git_info() -> dict:
    try:
        h = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": h, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--splits-dir", default=None, help="default: <data-dir>/splits")
    ap.add_argument("--out-root", default=str(ROOT), help="models go to <out-root>/models, metrics to <out-root>/outputs/training")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--subset", type=int, default=0, help="smoke test: use N train rows (and N/4 val and test)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--train-batch-size", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--force", action="store_true", help="overwrite an existing run")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.run_name: cfg["run_name"] = a.run_name
    if a.epochs: cfg["epochs"] = a.epochs
    if a.max_length: cfg["max_length"] = a.max_length
    if a.train_batch_size: cfg["train_batch_size"] = a.train_batch_size
    run = cfg["run_name"]
    splits_dir = Path(a.splits_dir) if a.splits_dir else Path(a.data_dir) / "splits"
    model_dir = Path(a.out_root) / "models" / run
    out_dir = Path(a.out_root) / "outputs" / "training" / run
    if (out_dir / "metrics.json").exists() and not a.force:
        log(f"{run}: metrics.json already exists, nothing to do (use --force to redo)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    use_amp = bool(cfg["fp16"]) and dev.type == "cuda"
    set_seed(cfg["seed"])
    log(f"run {run} | device {dev} | fp16 {use_amp} | max_length {cfg['max_length']}")

    frames, info, hashes = td.load(cfg["dataset"], a.data_dir, splits_dir, cfg)
    if a.subset:
        frames = td.subset_frames(frames, a.subset, cfg["seed"])
        info = {k: {**v, "subset_n": int(len(frames[k]))} for k, v in info.items()}
        log("SMOKE TEST subset sizes: " + str({k: len(v) for k, v in frames.items()}))
    log("split sizes: " + json.dumps({k: v["n"] for k, v in info.items()}))

    class_names = td.CLASS_NAMES[cfg["dataset"]]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    seq = {k: encode(tok, d["text"].tolist(), cfg["max_length"]) for k, d in frames.items()}
    y = {k: d["label"].to_numpy() for k, d in frames.items()}
    trunc = {k: float(np.mean([len(s) >= cfg["max_length"] for s in v])) for k, v in seq.items()}
    log("share of texts cut at max_length: " + json.dumps({k: round(v, 4) for k, v in trunc.items()}))
    train_majority = int(np.bincount(y["train"]).argmax())

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2).to(dev)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    bs, acc = cfg["train_batch_size"], cfg["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(seq["train"]) / (bs * acc))
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(cfg["seed"])
    pad = tok.pad_token_id

    history, best_f1, best_epoch, bad = [], -1.0, -1, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = rng.permutation(len(seq["train"]))
        t_ep, loss_sum, n_micro = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        micro = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        for mi, idx in enumerate(micro, 1):
            ids, att = collate([seq["train"][j] for j in idx], pad, dev)
            lab = torch.tensor(y["train"][idx], dtype=torch.long, device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss / acc).backward()
            loss_sum += float(loss.item()); n_micro += 1
            if mi % acc == 0 or mi == len(micro):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
        val_logits = predict_logits(model, seq["val"], cfg["eval_batch_size"], pad, dev, use_amp)
        vs = score(y["val"], val_logits.argmax(1))
        history.append({"epoch": ep, "train_loss": loss_sum / max(1, n_micro), "val_macro_f1": vs["macro_f1"],
                        "val_accuracy": vs["accuracy"], "epoch_seconds": round(time.time() - t_ep, 1)})
        log(f"epoch {ep}: " + json.dumps(history[-1]))
        if vs["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = vs["macro_f1"], ep, 0
            model.save_pretrained(model_dir / "best"); tok.save_pretrained(model_dir / "best")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"early stopping after epoch {ep} (best epoch {best_epoch})")
                break

    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    best = AutoModelForSequenceClassification.from_pretrained(model_dir / "best").to(dev)
    res = {}
    preds = {}
    for k in ("val", "test"):   # test is touched once, here, after the checkpoint is chosen
        lg = predict_logits(best, seq[k], cfg["eval_batch_size"], pad, dev, use_amp)
        res[k] = full_metrics(y[k], lg, train_majority, cfg["bootstrap_resamples"], cfg["seed"], class_names)
        preds[k] = lg
    man = td.load_manifest(splits_dir)
    meta = {"run_name": run, "config": cfg, "subset_smoke_test": a.subset or None, "git": git_info(),
            "seed": cfg["seed"], "split_id_hashes": man["id_hashes"], "source_sha256_checked": hashes,
            "data_info": info, "share_cut_at_max_length": trunc, "train_majority_class": train_majority,
            "class_names": class_names, "best_epoch": best_epoch, "history": history,
            "runtime_seconds": round(time.time() - T0, 1),
            "hardware": {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                         "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else None},
            "versions": {"torch": torch.__version__, "python": sys.version.split()[0]}}
    try:
        import transformers, sklearn
        meta["versions"].update({"transformers": transformers.__version__, "scikit_learn": sklearn.__version__})
    except Exception:
        pass
    (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "val": res["val"], "test": res["test"]}, indent=1))
    np.savez_compressed(out_dir / "predictions.npz", val_ids=np.array(frames["val"]["id"].astype(str)), val_labels=y["val"], val_logits=preds["val"],
                        test_ids=np.array(frames["test"]["id"].astype(str)), test_labels=y["test"], test_logits=preds["test"])
    t = res["test"]
    log(f"TEST macro-F1 {t['macro_f1']:.4f} (95% CI {t['bootstrap_95ci']['macro_f1'][0]:.4f}-{t['bootstrap_95ci']['macro_f1'][1]:.4f}), "
        f"accuracy {t['accuracy']:.4f}, majority baseline accuracy {t['majority_baseline']['accuracy']:.4f}")
    log(f"saved {out_dir}")


if __name__ == "__main__":
    main()
