# Thesis writing folder

Working drafts of the Master's thesis, structured after `Sample_Master_Thesis.pdf`
(same front-matter order, numbering depth, citation style). The official template
comes later; these files are plain Markdown so they can be poured into Word or
LaTeX without rework.

## Files (in reading order)

| File | Content | Sample equivalent |
|------|---------|-------------------|
| `00_front_matter.md` | Title page, Erklärung, AI declaration, acknowledgement, abstract | pp. i-3 |
| `01_introduction.md` | Motivation, problem, aim, RQs, structure | Ch. 1 |
| `02_background.md` | Literature review, research gap | Ch. 2-3 |
| `03_methodology.md` | Pipeline, IG, mutation, tracks, metrics | Ch. 2.x / 4 |
| `04_experimental_setup.md` | Data, fine-tuning, hyperparameters, hardware | Ch. 5 |
| `05_results_discussion.md` | RQ1-RQ3 results, statistics, discussion | Ch. 6 |
| `06_conclusion.md` | Findings, limitations, applicability, future work | Ch. 7 |
| `07_appendix.md` | Extra tables, configs, reproducibility | Appendices |
| `references.bib` | The 18 references from the proposal | References |

## Conventions taken from the sample

- Headings numbered up to three levels (1 / 1.1 / 1.1.1). Numbering is added at build time (`--number-sections`), so do not type numbers into headings.
- Citations author-year, e.g. `(Sundararajan et al. 2017)`. Write `[@sundararajan2017]` and let the build produce it.
- Figures and tables get a numbered caption, referenced in the text before they appear; lists of figures and tables are generated.
- URLs and tools go in footnotes with an access date, e.g. `^[https://captum.ai (Accessed: 13th October 2026)]`.
- Front matter order: title page, Erklärung, declaration on AI use, acknowledgement, abstract, contents, list of figures, list of tables.
- Target length: the sample is about 118 pages; ask the supervisors for the required range.

## Rules for myself

- Text is reused from the proposal where marked `[REUSE: proposal §x]`; rewrite it in past tense once the code is fixed.
- Every number in the thesis comes from `outputs\` through a script, never typed by hand.
- Anything in `[[double brackets]]` is a placeholder to fill in. Search for `[[` before submission.

## Draft build (needs pandoc)

```
pandoc 00_front_matter.md 01_introduction.md 02_background.md 03_methodology.md ^
  04_experimental_setup.md 05_results_discussion.md 06_conclusion.md 07_appendix.md ^
  --citeproc --bibliography references.bib --number-sections --toc ^
  -o thesis_draft.docx
```

The result is a rough draft for reading, not the final layout. Lists of figures and tables come from the official template (Word or LaTeX), not from this draft build.
