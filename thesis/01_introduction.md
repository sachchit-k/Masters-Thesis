# Introduction and Motivation



## Motivation

Automated systems that flag disinformation have become part of how online platforms moderate content, and the models behind them are increasingly fine-tuned transformer encoders. At the same time, the rules governing such systems are tightening. The European Union's Artificial Intelligence Act sets requirements on accuracy, robustness and cybersecurity for high-risk AI systems in its Article 15 [[cite: Regulation (EU) 2024/1689; add to references.bib]], and it asks for transparency towards those who deploy and oversee such systems. A detector that reports a high score on a benchmark is therefore not yet a detector that can be shown to be reliable.



The concern is that high scores can hide how the decision is reached. Deep networks tend to minimise training error by picking up the easiest regularities in the data, including artefacts that happen to correlate with the label but have nothing to do with the task. This behaviour is known as shortcut learning [@geirhos2020]. For disinformation detection, the likely shortcuts are lexical: sensational words, emotive adjectives, particular punctuation or formatting that is typical of one source type in the training data rather than of false content as such [@ma2025; @wan2025]. A model that has learned such cues can score well on a test split drawn from the same distribution and still fail when the cues are absent or reversed. Evidence from outside a single benchmark supports the worry: averaged over 16 misinformation benchmarks, standard language-model-based and LLM-based detectors reach only 58.8 to 80.2 percent accuracy [@wan2025], so a strong number on one dataset does not carry over.

This dependence has two practical consequences, which together form a security gap. A person who wants to spread false content can launder it through a neutral, formal register that avoids the trigger words, and the detector may let it through. In the other direction, accurate text that is written in an emotional style may be flagged wrongly. Both failure modes are invisible to accuracy and macro-F1, because those metrics average over the cases the model gets right for the wrong reasons.

## Problem Statement and Research Gap

Much of the existing work on text perturbation treats word substitution as an attack: the aim is to change the prediction with as little change to the text as possible, using black-box search or auxiliary selector models to find the words to replace [@jin2020; @li2020; @hossam2021; @zhang2020]. That is a useful way to measure worst-case fragility, but it is not an audit. It does not say which words the model actually relies on, and it does not separate the loss caused by removing a genuine trigger from the loss caused by simply damaging the sentence.

Post-hoc explanation methods give a way to ask the first question. Sampling-based methods such as LIME and SHAP approximate a model's local behaviour from perturbed copies of the input [@ribeiro2016; @lundberg2017]. Integrated Gradients (IG) instead uses the model's own gradients and satisfies the sensitivity and completeness axioms [@sundararajan2017], which makes its output deterministic for a fixed configuration. Comparative studies of explanation techniques for text classifiers already exist [@atanasova2020]. To the best of my knowledge, however, there is no study that uses IG token attributions to steer controlled, semantics-preserving perturbations and reads the result as a diagnostic of shortcut learning. 

This thesis addresses three gaps that follow from this picture; the full argument is given in Chapter 2.

1. **No deterministic, white-box attribution audit.** Existing attacks locate influential words with black-box procedures, so they say little about what the encoder's inner layers use.
2. **No comparison across text registers.** How concentrated the trigger words are in short statements versus long articles has not been profiled [@verma2021; @wang2017].
3. **No structural control baselines.** Without a Least-Relevant-First (LeRF) track and a random baseline next to the Most-Relevant-First (MoRF) track, a drop in accuracy cannot be traced to the removal of important words rather than to a general loss of grammar and fluency [@samek2017].



## Aim, Objectives and Research Questions

**Aim.** The aim of this thesis is to design, implement and evaluate an attribution-guided diagnostic framework that quantifies shortcut learning and the influence of trigger words in transformer-based disinformation detectors, and to compare short-form and long-form text. The framework fine-tunes RoBERTa-base classifiers on LIAR (short political statements) [@wang2017] and WELFake (full news articles) [@verma2021], extracts word-level Integrated Gradients attributions with Captum [@kokhlikyan2020], and then mutates words in order of attribution with a masked language model under a semantic similarity constraint. Three tracks (MoRF, LeRF and Random) are compared, so that the effect of targeting important words can be separated from the effect of editing in general.

**Objectives.** 

1. Review transformer-based disinformation detection and the post-hoc explanation methods used to audit it, with a focus on evidence of shortcut learning and trigger-word reliance.
2. Establish a deterministic attribution pipeline based on Integrated Gradients, with a defined way of aggregating sub-word scores into word-level scores.
3. Evaluate classifier stability with a label-preserving mutation pipeline that replaces the selected words using a masked language model, under part-of-speech and semantic-similarity constraints.
4. Examine how text register and length relate to the concentration of high-attribution words and to the resulting loss in classification performance.

**Research questions.**

- RQ1: How do variations in text register length affect the spatial concentration of high-weight decision tokens extracted via Integrated Gradients?
- RQ2: To what degree does an attribution-guided token mutation under a strict semantic invariance constraint cause a statistically greater drop in classification accuracy compared to Least-Relevant-First (LeRF) control baselines?
- RQ3: What are the limitations of the Integrated Gradients algorithm, specifically regarding baseline tensor selection and sub-word aggregation methods, in accurately quantifying contextual feature attributions?



## Contributions

The thesis is intended to contribute the following. 

- A diagnostic framework that connects deterministic Integrated Gradients attributions to semantics-constrained word replacement, so that explanation output is tested rather than only inspected.
- A control design with MoRF, LeRF and Random tracks that mutate the same number of words, which makes it possible to tell targeted loss from general degradation.
- A comparison of attribution concentration and mutation sensitivity between a short-form (LIAR) and a long-form (WELFake) register.
- An analysis of how sensitive the resulting word rankings are to the IG baseline and to the way sub-word scores are aggregated.



## Structure of the Thesis



Chapter 2 gives the background: shortcut learning and dataset artefacts, post-hoc attribution methods with Integrated Gradients in detail, text perturbation and semantic preservation, and related work, ending with the research gap. Chapter 3 describes the methodology: the pipeline, attribution extraction and aggregation, the mutation engine and its filters, the three evaluation tracks, and the metrics and statistical tests. Chapter 4 sets out the experimental setup: the two datasets and their splits, the fine-tuning of the target models, the settings used, and the computing environment. Chapter 5 presents and discusses the results for the three research questions. Chapter 6 concludes with a summary of the findings, the limitations of the study and directions for future work.