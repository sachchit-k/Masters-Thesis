# Results and Discussion

<!-- Skeleton only. Numbers come from outputs\ through scripts that write figures and LaTeX/Markdown tables. -->

## Baseline Model Performance

[[Macro-F1 and accuracy of the fine-tuned models on LIAR and WELFake.]]

## RQ1: Attribution Concentration Across Registers

[[Distribution of attribution mass over words; short vs. long text; figure and table.]]

## RQ2: Effect of Guided Mutation

### Accuracy Drop

[[MoRF vs. LeRF vs. Random per dataset, curve over mutation budget.]]

### Logit Shift

[[Same comparison on ground-truth-class logits.]]

### Statistical Significance

[[Paired Wilcoxon tests with effect sizes and corrected p-values.]]

## RQ3: Limitations of Integrated Gradients

### Baseline Sensitivity

[[Spearman correlation between rankings under zero and custom baselines.]]

### Aggregation Sensitivity

[[Rank distortion from signed vs. alternative aggregation.]]

## Qualitative Examples

[[A few mutated samples per track (keep short and only from non-sensitive examples).]]

## Discussion

[[What the results say about shortcut reliance per register; where the hypothesis fails; comparison with related work.]]

## Threats to Validity

[[Internal, external, construct validity.]]
