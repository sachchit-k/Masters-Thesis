# Methodology

<!-- Sample Ch. 2 "Research methodology" explains design first, then each method in a numbered section.
     [REUSE: proposal §3]. Write in past tense once code is fixed. Every decision here must match DECISIONS.md. -->

## Research Design

<!-- Experimental, quantitative, two registers, three tracks. One paragraph plus a mapping table
     (the sample has "Methodological Mapping Matrix"): RQ -> stage -> method -> metric -> output. -->

[[Draft and Table: RQ-to-method mapping]]

## Pipeline Overview

<!-- [REUSE: proposal Figure 1]. Redraw as a clean figure. -->

[[Figure: pipeline architecture]]

## Target Models

[[RoBERTa-base fine-tuned separately on LIAR and WELFake; what "optimised classifier" means; see Ch. 4 for settings.]]

## Stage 1: Attribution Extraction

### Integrated Gradients

<!-- Captum, embedding-layer attribution, 50 Riemann steps, internal batch size, target class,
     special tokens. DECISIONS.md items 5-8. -->

[[Draft]]

### Sub-word to Word Aggregation

<!-- Signed sum of sub-token scores to word level. DECISIONS.md item 9. -->

[[Draft]]

### Baseline and Aggregation Sensitivity (RQ3)

<!-- Zero vs. custom reference baseline; signed sum vs. alternatives; Spearman rank correlation. -->

[[Draft]]

## Stage 2: Guided Mutation

### Candidate Generation

<!-- BERT-base-uncased MLM, top-50 predictions. -->

[[Draft]]

### Constraint Filtering

<!-- POS alignment and Sentence-Transformer cosine similarity >= 0.85. State which models, which POS tagger.
     What happens when no candidate passes (DECISIONS.md item 11). -->

[[Draft]]

### Mutation Budget and Stopping Rule

<!-- Same number of mutated tokens on all tracks (DECISIONS.md items 10 and 12). -->

[[Draft]]

## Control Tracks

[[MoRF, LeRF, Random (10 seeds): definitions and why the controls are needed.]]

## Evaluation Metrics

<!-- [REUSE: proposal §3.5, Eq. 2-4]. Accuracy drop, logit shift on ground-truth class,
     Spearman attribution stability, attribution concentration measure for RQ1 (define it: e.g. share of total
     absolute attribution in the top-k words; decide in DECISIONS.md). -->

[[Draft]]

## Statistical Testing

<!-- Paired Wilcoxon signed-rank, pairing unit (DECISIONS.md item 14), effect size, multiple-comparison
     correction (item 15). -->

[[Draft]]

## Validity and Threats

[[Leakage, duplicates in WELFake, near-chance LIAR performance, MLM bias, similarity threshold choice.]]
