# CLAUDE.md - project rules

Thesis: "Trigger Words or True Evidence?" Audit of fine-tuned RoBERTa-base disinformation detectors (LIAR, WELFake) with Integrated Gradients + MLM-guided mutation (MoRF vs LeRF vs Random).

## Layout (Windows, drive D)
- `D:\Masters_Thesis_Sachchit\repo\`    Git repo: code, configs, thesis\, LOG.md, DECISIONS.md, .venv
- `..\data\`, `..\models\`, `..\outputs\` siblings of repo, never committed
- Caches live in `D:\cache` (HF_HOME, TORCH_HOME, PIP_CACHE_DIR, UV_CACHE_DIR). Never let anything write caches to C:.

## Hardware limits (GTX 1650 Ti, 4 GB VRAM, 24 GB RAM)
- fp16 (autocast / `.half()`) for training and inference; small batches + gradient accumulation; gradient checkpointing for WELFake at 512 tokens.
- Captum: always pass `internal_batch_size` of 5-10 (50 IG steps). Never the default.
- MLM (BERT-base) and Sentence-Transformer: inside `torch.no_grad()`; move to CPU if VRAM is tight.
- Free GPU memory between stages (`del`, `torch.cuda.empty_cache()`).

## Data and tensors
- Never read, print or log dataset rows or full tensors. Report shapes, counts, summary stats only.
- Save intermediates (attributions, mutated texts, logits) to `outputs\` so IG is never recomputed. Check for an existing file before computing.
- Every number in the thesis comes from `outputs\` through a script.

## Method fixed by the proposal
50 IG steps, sub-word to word aggregation by sum, POS match + cosine >= 0.85, 10 random seeds, all tracks mutate the same number of tokens per sample. Open choices live in DECISIONS.md; do not fill its rows unless asked.

## Working style
- Keep answers concise. Give code as files, not long pasted blocks.
- Ask before big or irreversible actions and before installing anything large or changing system settings.
- Never delete files on system drives (C:).
- Flag methodology risks (leakage, WELFake duplicates, near-chance LIAR accuracy).
- LOG.md: append-only, newest at bottom, 2-4 lines, tag each entry [code] or [doc], update the Status line.
- Git: commit code, configs, thesis\, LOG.md, DECISIONS.md. Never data, models, outputs, .venv.
- Git remote: `origin` = github.com/sachchit-k/Masters-Thesis (branch `main`). Git user.name/email are set per repo (not --global). Credentials sit in the Windows profile on C: (known exception to Rule 0).
