# Writing flow: documentation runs in parallel with the code

Rule: no code step is finished until its thesis text and layout items are updated. Each code step has a documentation gate (section 3). The target layout is the one in `Sample_Master_Thesis.pdf` until the official template replaces it.

## 1. Flow of a thesis (from idea to submission)

| Phase | What happens | Output |
|-------|--------------|--------|
| A. Frame | Fix title, RQs, objectives, chapter outline, citation style, template choice | Proposal (done), thesis\ skeleton (done) |
| B. Ground | Read and cite literature; keep notes per source; write Chapter 2 as you read | Chapter 2 draft, references.bib growing |
| C. Method | Write each method section when its decision is made, in the same sitting as the DECISIONS.md entry | Chapter 3 draft, always in sync with DECISIONS.md |
| D. Setup | Write data, model, settings and hardware facts straight from configs, logs and run results | Chapter 4 draft |
| E. Results | Generate figures and tables by script; write what they show, no interpretation yet | Chapter 5 results part |
| F. Interpret | Answer each RQ, compare with related work, state limits | Chapter 5 discussion, Chapter 6 |
| G. Frame again | Introduction (rewritten to match what was done), conclusion, abstract, acknowledgement, declarations | Chapter 1, front matter |
| H. Revise | Passes in this order: structure, argument, consistency, language, layout, citations | Revised full draft |
| I. Feedback | Send draft to supervisors, log their comments, resolve each one | Tagged draft versions |
| J. Submit | Final template, checks, print or upload | Final PDF, Git tag |

Chapters 2 and 3 can be written early because they do not depend on results. Chapters 1 and 6 and the abstract are written last because they must match the results.

## 2. Parallel plan: code step to chapter

| Code step (PROJECT_STEPS.md) | Thesis section to write or update | Layout items due |
|---|---|---|
| 2 Repo, environment, smoke test | Chapter 4: computing infrastructure, software versions | Software versions table started |
| 4 Data preparation | Chapter 4: datasets (LIAR, WELFake), splits, duplicate handling | Table: dataset characteristics; figure: length distributions |
| 5 Fine-tuning | Chapter 4: fine-tuning settings, baseline performance; Chapter 5: baseline model performance | Table: hyperparameters; table: macro-F1 and accuracy |
| 6 Attribution (IG) | Chapter 3: IG, baseline, aggregation; Chapter 5: RQ1 | Equation numbering; figure: attribution concentration per register |
| 7 Sensitivity | Chapter 3: sensitivity design; Chapter 5: RQ3 | Table: Spearman correlations |
| 8 Mutation engine | Chapter 3: candidate generation, filters, budget, no-candidate policy | Figure: pipeline (redrawn); table: filter statistics |
| 9 Experiment runs | Chapter 5: RQ2 results | Figures: MoRF, LeRF, Random curves; example table |
| 10 Statistics | Chapter 3: statistical testing; Chapter 5: significance | Table: Wilcoxon results with effect sizes |
| Throughout | Chapter 2: add each source when first cited | references.bib entry per citation |

## 3. Documentation gate (after every code step)

Do all five before saying a step is done:
1. Update the chapter section from the table above (even a rough paragraph; mark gaps with `[[ ]]`).
2. Add new figures and tables via script into `thesis\figures` and `thesis\tables`, with caption text and a label.
3. Add any new source to `references.bib` and cite it as `[@key]`.
4. Sync `DECISIONS.md` and Chapter 3 or 4 (same wording for the same choice).
5. Append a `[doc]` entry to `LOG.md`: what was written, what is still a placeholder.

## 4. Layout checklist (from the sample thesis)

Front matter order: title page, Erklärung, Declaration on the use of AI, Acknowledgement, Abstract, (Zusammenfassung if required), Table of Contents, List of Figures, List of Tables.
- Page numbers: the sample starts numbering at the AI declaration page, the title page and Erklärung are unnumbered. Confirm with the official template.
- Headings numbered to three levels (1, 1.1, 1.1.1); front matter and appendix headings unnumbered.
- Figures: numbered consecutively, caption below, referenced in the text before they appear, source named if not your own. Tables: numbered consecutively, caption as in the sample (check pp. 13-14 for position), referenced in the text. Both listed in the lists of figures and tables.
- Citations: author-year in the text, full list in References, sorted alphabetically. Every cited work is in the list and the reverse.
- URLs and tools: footnote with "(Accessed: day month year)".
- Equations numbered and explained (all symbols defined at first use).
- Abbreviations written out at first use; optional list of abbreviations.
- Appendices after References: hyperparameters, extra results, decision log, reproducibility.
- Language: one variety of English throughout; past tense for what was done, present tense for what the thesis does or shows.
- Length: sample is about 118 pages; confirm the required range with the supervisors.

## 5. Support files to keep in thesis\

- `notation.md`: every symbol and term with one definition (Δz, ΔAccuracy, MoRF, LeRF, word score, budget k). Use the same wording everywhere.
- `figures\` and `tables\`: generated only; never edit by hand.
- `feedback.md`: dated supervisor comments with status (open, resolved, rejected with reason).
- `notes\`: per-source reading notes (claim, method, dataset, limit, usable quote with page). Not part of the thesis.

## 6. Weekly rhythm

- Start of week: read LOG.md status; pick the code step and the matching sections.
- During the week: write the method or setup text the same day the code choice is made.
- End of week: run the draft build (README.md), read the newly written pages once, fix layout items, update LOG.md.
- Before each supervisor meeting: tag the draft (`git tag draft-YYYY-MM-DD`) and send the built file.

## 7. Revision passes (phase H)

1. Structure: each chapter does its job; every RQ answered where Chapter 1 says it will be.
2. Argument: each claim has evidence (own result or citation); no result is interpreted before it is shown.
3. Consistency: numbers match scripts' outputs; notation, terminology and abbreviations are uniform; DECISIONS.md matches the text.
4. Language: short sentences, no unexplained jargon, no unsupported superlatives.
5. Layout: checklist in section 4, then placeholder search (`[[` and `[REUSE`).
6. Citations: every in-text citation resolves; every reference is cited; DOIs and URLs work.
7. Declarations: Erklärung and AI-use declaration are final and truthful.
