# Experimental Setup

<!-- Facts only; this chapter is mostly filled from DECISIONS.md, the run log and the configs. -->

## Datasets

### LIAR (Short-Form)

[[Source, size, label scheme used (binary or six-way), splits, average length, preprocessing.]]

### WELFake (Long-Form)

[[Source, size, title + body handling, duplicate and leakage handling, splits, length distribution, truncation at 512 tokens.]]

### Table: Dataset Characteristics

[[Table: size, classes, balance, mean tokens, splits per dataset]]

## Fine-Tuning the Target Models

[[RoBERTa-base, optimiser, learning rate, epochs, batch size, gradient accumulation, fp16, seeds, early stopping; held-out macro-F1 and accuracy per dataset. Note if LIAR is near chance.]]

## Attribution and Mutation Settings

[[IG settings, baseline definitions, MLM and Sentence-Transformer model names and versions, POS tagger, thresholds, budgets, random seeds.]]

## Computing Infrastructure

<!-- From the run log. Hardware: Acer Nitro AN515-55, i5-10300H, 24 GB RAM, GTX 1650 Ti (4 GB VRAM), Windows 11.
     Add library versions (PyTorch, Transformers, Captum) and total runtime per stage. -->

[[Draft]]

## Reproducibility

[[Repository, commit hash, how to rerun, what is not published (data, checkpoints).]]
