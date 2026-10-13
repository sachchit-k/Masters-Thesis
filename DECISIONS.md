# Methodology Decisions
Settle each before writing the code it affects. Record date and reason. Status: OPEN or DECIDED.

| # | Decision | Status | Choice | Reason | Date |
|---|----------|--------|--------|--------|------|
| 1 | Train/val/test splits (LIAR, WELFake) | OPEN | | | |
| 2 | Global seed policy | OPEN | | | |
| 3 | WELFake duplicate and leakage handling | OPEN | | | |
| 4 | Clean baseline: accuracy and F1 per dataset (note if LIAR is near chance) | OPEN | | | |
| 5 | IG baseline input (e.g. all-[PAD], keep [CLS]/[SEP]) | OPEN | | | |
| 6 | IG target class (predicted vs true) | OPEN | | | |
| 7 | IG settings: 50 Riemann steps, internal_batch_size 5-10 | OPEN | | | |
| 8 | Special tokens and punctuation in attribution | OPEN | | | |
| 9 | Sub-word to word aggregation (sum) | OPEN | | | |
| 10 | Mutation budget (tokens or fraction per sample) and stopping rule | OPEN | | | |
| 11 | No candidate passes POS and cosine >= 0.85 filter | OPEN | | | |
| 12 | Same number of mutated tokens across MoRF, LeRF, Random | OPEN | | | |
| 13 | Random track: 10 seeds | OPEN | | | |
| 14 | Wilcoxon pairing unit | OPEN | | | |
| 15 | Multiple-comparison correction | OPEN | | | |
| 16 | Max sequence length (512 for WELFake) and VRAM settings (fp16, batch, accumulation) | OPEN | | | |

## Notes
- 2026-10-13 Tooling: all code and Git commits are done in Cursor with Claude Code integrated; CLAUDE.md in repo\ holds the rules Claude Code follows. Process decision, not one of items 1-16.
- 2026-10-13 Tooling: Git remote is GitHub (sachchit-k/Masters-Thesis, branch main); first commit 6cce445. Process decision, not one of items 1-16.
