# Introduction and Motivation

<!-- Sample Ch. 1: no subsection stack, flowing prose. Here the proposal's sections 1.1-1.2 give a head start.
     Aim for 4-6 pages. -->

## Motivation

<!-- [REUSE: proposal §1.1] Shortcut learning in transformer detectors; EU AI Act Article 15 (accuracy,
     robustness, transparency); security gap (laundering disinformation through neutral registers,
     false positives on emotive but true text). -->

[[Draft]]

## Problem Statement and Research Gap

<!-- [REUSE: proposal §2.2] Three gaps: (1) no intrinsic deterministic attribution audit,
     (2) no cross-register and scale profiling, (3) no structural control baselines (LeRF, Random).
     Keep the short version here, full argument goes to Chapter 2. -->

[[Draft]]

## Aim, Objectives and Research Questions

<!-- [REUSE: proposal §1.2 and §2.3] -->

**Aim.** [[Overall aim]]

**Objectives.** [[Objectives 1-4, rewritten to match what was actually done]]

**Research questions.**

- RQ1: How do variations in text register length affect the spatial concentration of high-weight decision tokens extracted via Integrated Gradients?
- RQ2: To what degree does an attribution-guided token mutation under a strict semantic invariance constraint cause a statistically greater drop in classification accuracy compared to Least-Relevant-First (LeRF) control baselines?
- RQ3: What are the limitations of the Integrated Gradients algorithm, specifically regarding baseline tensor selection and sub-word aggregation methods, in accurately quantifying contextual feature attributions?

<!-- If the final research questions change during the work, update here and in Ch. 5 and Ch. 6 together. -->

## Contributions

[[2-4 bullet points of what is new: diagnostic framework, MoRF/LeRF/Random control design, cross-register profile, IG limitation analysis.]]

## Structure of the Thesis

<!-- One paragraph per chapter, written last. -->

[[Draft]]
