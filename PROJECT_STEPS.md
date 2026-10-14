# PROJECT STEPS (paste into project instructions)

Thesis: "Trigger Words or True Evidence? Auditing Disinformation Detectors (Encoder Models) with Attribution-Guided Perturbation"
Folder: D:\Masters_Thesis_Sachchit  (repo\ = Git, data\, models\, outputs\ never committed)
Read at the start of every session: repo\LOG.md (status line and last entries), repo\DECISIONS.md, repo\CLAUDE.md, repo\PROJECT_STEPS.md, repo\thesis\WRITING_FLOW.md.

## Working rules (apply to every step)

1. One step at a time. Do the step, check it, log it, then say what the next step is. Do not start the next step unasked.
2. Log every step in repo\LOG.md: append-only, newest at the bottom, 2-4 lines per entry, tag [code] or [doc], add "Lost time:" only when something cost time. Update the Status line.
3. Record each methodology choice in repo\DECISIONS.md (choice, reason, date, status DECIDED) before writing the code it affects. Never fill rows without my agreement.
4. Ask before big or irreversible actions (large downloads, installs, deleting, history rewrites). Never delete files on system drives. C: is nearly full: everything heavy goes to D:.
5. Never read or print dataset rows or full tensors. Print shapes, counts and summary statistics only. Save intermediates (attributions, mutated texts, logits) to outputs\ so IG is never recomputed.
6. Hardware limits (GTX 1650 Ti, 4 GB VRAM, 24 GB RAM): fp16, small batches, gradient accumulation and checkpointing for WELFake at 512 tokens. Captum IG always with internal_batch_size 5-10. MLM and Sentence-Transformer under torch.no_grad(), on CPU if VRAM is tight.
7. Flag methodology risks as soon as seen: leakage between splits, duplicates in WELFake, near-chance LIAR accuracy, unequal mutation counts across tracks, small test sets.
8. Keep answers concise. Give code as files, not long pasted blocks.
9. Every number in the thesis comes from outputs\ through a script. Nothing typed by hand.
10. Commit small and often. Commit messages say what and why. Never commit data\, models\, outputs\, .venv or secrets.
11. Documentation runs in parallel: after every code step do the documentation gate in repo\thesis\WRITING_FLOW.md before calling the step done.
12. AI use is declared truthfully in the thesis. Keep a short list in LOG.md of what Claude was used for.

## Status of steps

| Step | Name | State |
|------|------|-------|
| 1 | Machine setup | DONE |
| 2 | Repo, environment, smoke test | NEXT |
| 3 | Documentation setup (thesis\ skeleton) | DONE (not yet committed) |
| 4 | Data preparation and first decisions | pending |
| 5 | Fine-tune target models | pending |
| 6 | Attribution extraction (IG) | pending |
| 7 | Attribution sensitivity (RQ3) | pending |
| 8 | Mutation engine | pending |
| 9 | Experiment runs (MoRF, LeRF, Random) | pending |
| 10 | Statistics, figures, tables | pending |
| 11 | Writing track (runs alongside 4-10) | in progress |
| 12 | Final checks and submission | pending |

Schedule check: the proposal plans fine-tuning in September and IG in October. Code work has not started yet, so the project is about one month behind that plan. Say so in the log and re-plan the timeline in Step 4 rather than ignoring it.

---

## Step 1: Machine setup (DONE)

Caches and temp redirected to D:\cache (HF_HOME, TORCH_HOME, PIP, UV, NLTK, CUDA, MPL, XDG, npm). GPU visible, C: 58 GB free, sleep disabled on AC. Old July prototype (D:\2026-UNI-KOB\thesis\Code) is an archive, not reused.

## Step 2: Repo, environment, smoke test

Goal: a clean, reproducible base before any real work.
Tasks:
1. Write repo\CLAUDE.md from the working rules above (project facts, layout, hardware limits, rules).
2. Create the environment on D: (.venv inside repo\, uv or pip with caches on D:). Install PyTorch with CUDA, Transformers, Captum, sentence-transformers, spaCy or NLTK (POS tagger), scipy, statsmodels, pandas, scikit-learn, matplotlib. Pin versions in requirements.txt (or pyproject plus lock file). Ask before large installs.
3. Write src\ and tests\ folders, a config\ folder (YAML for paths, seeds, model names) and a paths module that reads the D: layout so no path is hardcoded elsewhere.
4. Smoke test (no real datasets): GPU visible, RoBERTa-base loads in fp16, forward pass on a dummy sentence, one Captum IG pass with internal_batch_size set, BERT-base-uncased MLM loads and predicts a masked token, a Sentence-Transformer loads and returns a cosine similarity. Record peak VRAM.
5. Check .gitignore covers data\, models\, outputs\, .venv, caches, __pycache__, .env.
6. First commit: repo\ including thesis\, LOG.md, DECISIONS.md, CLAUDE.md, requirements, src skeleton.
Done when: smoke test passes from a fresh shell, git status is clean, nothing large is tracked.
Log: [code] entry with versions, VRAM numbers, commit hash.

## Step 3: Documentation setup (DONE)

repo\thesis\ has README, 00_front_matter to 07_appendix as Markdown, references.bib (18 proposal references, authors as initials). Structure follows Sample_Master_Thesis.pdf. Deferred: AI-use declaration, personal details (matriculation number, degree name, submission date, German abstract, page limit), final template.

## Step 4: Data preparation and first decisions

Goal: trustworthy data and splits.
Decide and record in DECISIONS.md first: #1 splits, #2 seed policy, #3 WELFake duplicate and leakage handling, #16 max length and VRAM settings. Also decide: LIAR labels (binary or six-way; the proposal quotes six-way baselines but never says which is used).
Tasks:
1. Download LIAR and WELFake into data\ (ask first, check size). Record source, version, licence, download date and file hashes in LOG.md.
2. Write a loader that returns splits without printing text. Check class balance, length statistics (tokens, words), missing values.
3. WELFake: detect exact and near duplicates (hashing, then MinHash or embedding similarity) within and across splits. Apply the decided handling. Report counts only.
4. Freeze splits to files (ids only) in data\splits\ with the seed recorded.
5. Re-plan the timeline against today's date and the submission deadline and log it.
Done when: split files exist, duplicate report exists, DECISIONS #1-3 and #16 are DECIDED.

## Step 5: Fine-tune target models

Goal: two optimised RoBERTa-base classifiers, one per dataset.
Tasks:
1. Training script with config file, fp16, small batch, gradient accumulation, early stopping on validation macro-F1, fixed seeds. WELFake at 512 tokens with gradient checkpointing if needed.
2. Train on LIAR, then WELFake. Save checkpoints to models\ and metrics to outputs\.
3. Evaluate on the held-out test split: accuracy, macro-F1, confusion matrix. Note if LIAR is near chance (flag it: attributions on a near-chance model are hard to interpret) and decide with me whether to continue, change label scheme or note it as a limitation.
4. Record DECISIONS #4 (clean baseline).
Done when: both checkpoints load and reproduce the logged metrics.
Log: [code] metrics, runtime, VRAM, any lost time.

## Step 6: Attribution extraction (IG)

Goal: word-level attribution for every evaluation sample, computed once.
Decide first: #5 baseline input, #6 target class, #7 IG settings (50 Riemann steps, internal_batch_size 5-10), #8 special tokens and punctuation, #9 sub-word to word aggregation.
Tasks:
1. Implement IG on the embedding layer with Captum. Map sub-tokens to words, aggregate by signed sum, drop or mask special tokens as decided.
2. Sanity checks on a small subset: completeness (sum of attributions close to the output difference, report the convergence delta), determinism (two runs identical), effect of step count.
3. Run on the full evaluation set in resumable batches. Save per-sample word scores, ranks and metadata to outputs\attributions\. Never recompute.
4. RQ1 measure: define attribution concentration (for example share of total absolute attribution in the top-k words, entropy, Gini) and decide it in DECISIONS.md.
Done when: attributions exist for every sample of both datasets, checks logged.

## Step 7: Attribution sensitivity (RQ3)

Goal: show how much the word ranking depends on IG configuration.
Tasks: repeat Step 6 on a fixed subset with (a) the zero baseline and a custom reference baseline (for example all-PAD or mask), (b) signed sum versus alternative aggregations (for example absolute sum, L2 norm, mean). Compute Spearman rank correlation between rankings, per sample and aggregated. Save to outputs\sensitivity\.
Done when: correlation tables exist for both datasets.

## Step 8: Mutation engine

Goal: label-preserving mutation that is deterministic given a seed.
Decide first: #10 mutation budget and stopping rule, #11 behaviour when no candidate passes, #12 equal number of mutated tokens across tracks.
Tasks:
1. MLM candidate generation (BERT-base-uncased, top 50 predictions for the masked position, in context).
2. Filters: POS match with the original word, then Sentence-Transformer cosine similarity >= 0.85 between original and mutated text. Exclude the original word and sub-word fragments.
3. Unit tests on synthetic sentences: filters reject what they should, budget is respected, seeds reproduce results.
4. Log the share of words with no passing candidate per dataset (it may bias MoRF vs LeRF if it differs by track).
Done when: tests pass and a dry run on a small subset gives sensible counts.

## Step 9: Experiment runs

Goal: the full MoRF, LeRF and Random comparison.
Tasks:
1. For each sample and each budget level, mutate the same number of words on every track. MoRF in descending attribution order, LeRF ascending, Random with 10 seeds.
2. Re-run the classifier on all mutated texts. Save predictions and ground-truth-class logits (Δz) per sample, track, budget and seed to outputs\results\. Save mutated texts too, so nothing is recomputed.
3. Run on a small subset first, check timing and VRAM, then scale. Resumable batches with checkpoints.
Done when: result files cover all samples, tracks and seeds, and a completeness check passes (counts match).

## Step 10: Statistics, figures and tables

Decide first: #14 Wilcoxon pairing unit, #15 multiple-comparison correction.
Tasks:
1. Accuracy drop (A_orig minus A_mut), logit shift Δz, per track and dataset.
2. Paired Wilcoxon signed-rank: MoRF vs LeRF and MoRF vs Random, effect sizes, corrected p-values.
3. RQ1 comparison across registers, RQ3 correlation tables.
4. Scripts write figures (PDF) and tables to repo\thesis\figures and repo\thesis\tables so the thesis updates by rerunning.
Done when: every result table and figure the thesis needs is generated by a script.

## Step 11: Writing track (runs in parallel with Steps 2-10)

Full flow, chapter-to-step mapping and layout checklist: repo\thesis\WRITING_FLOW.md. Follow it.
Core rule: a code step is not done until its documentation gate is done (update the chapter section, add generated figures and tables, add sources to references.bib, sync DECISIONS.md with the text, append a [doc] entry to LOG.md).
Order: Chapters 2 and 3 early (they do not depend on results), Chapter 4 as facts become known, Chapter 5 as results arrive, Chapters 1 and 6, abstract and declarations last.
Send a full draft to the supervisors well before the deadline (ask for their preferred lead time); tag drafts in Git.

## Step 12: Final checks and submission

1. Reproducibility: fresh clone plus requirements reproduces a small run; record commit hash and environment in Appendix D.
2. Consistency: notation (ΔAccuracy, Δz, MoRF, LeRF), RQ answers match Chapters 1, 5 and 6, all figures and tables referenced.
3. Search for leftover placeholders ([[ and [REUSE).
4. Fill declaration of authorship (Erklärung), the declaration on the use of AI (truthful, specific), acknowledgement, personal details.
5. Convert to the official template, check page limit, fonts, margins, lists of figures and tables, citation style.
6. Submit as the examination office requires; keep a tagged Git release of the final state.

---

## Decisions checklist (DECISIONS.md), by step

Step 4: #1 splits, #2 seed policy, #3 WELFake duplicates and leakage, #16 max length and VRAM settings.
Step 5: #4 clean baseline.
Step 6: #5 IG baseline input, #6 target class, #7 IG settings, #8 special tokens, #9 aggregation.
Step 8: #10 mutation budget and stopping rule, #11 no-candidate policy, #12 equal mutation count, #13 Random 10 seeds.
Step 10: #14 Wilcoxon pairing unit, #15 multiple-comparison correction.

## Deferred by me (do not ask again until Step 12 unless a step needs it)

AI-use declaration wording, title-page details, German abstract, page limit, final template.
