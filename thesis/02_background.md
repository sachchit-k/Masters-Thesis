# Background and Literature Review



## Literature Search Approach



[[Draft]]***This chapter presents the research design and methodological framework constructed to achieve the objectives of this thesis.***

## Disinformation Detection with Transformer Encoders



### Task Definition and Benchmarks



Fine-tuned transformer encoders are the dominant architecture for automated disinformation detection, though reported performance varies sharply with label granularity and text register. On short-form political statements, classical baselines (Logistic Regression, SVM, CNN, Bi-LSTM) top out around 27% accuracy on LIAR's six-way classification task [@wang2017liar], barely above the 17% random baseline, while fine-tuned BERT reaches 97.8% on the binary Kaggle Fake News benchmark [@jin2020bert]. On long-form and multimodal news, recent work pushes past 90%: DICE, a causality-aware graph network built on BERT and ResNet backbones, reaches 87 to 93% accuracy across Twitter, Weibo, Fakeddit and Pheme [@ma2024graphing], and a hybrid-attention LLM feature-fusion model reports an F1 of 0.945 on WELFake [@xu2025hybrid]. Cross-benchmark evaluation complicates this picture: averaged across 16 misinformation benchmarks, standard LM-based detectors (BERT, DeBERTa) and LLM-based detectors (Llama3) score only 58.8 to 80.2% accuracy [@wan2025truth], which shows that strong single-benchmark numbers do not guarantee robust, generalisable detection.



The two registers used in this thesis are the following [as in proposal section 3.2]:

- **Short-form (LIAR [@wang2017liar]):** brief political statements averaging 17.9 tokens per claim, a sparse register in which classifiers are expected to be vulnerable to localised keyword triggers.
- **Long-form (WELFake [@verma2021welfake]):** full news articles combining title and body text, averaging several hundred words per document and so considerably longer than LIAR's 17.9-token statements; a multi-sentence register with extended contextual dependencies.



### Fine-Tuned Encoders: BERT and RoBERTa



[[Draft: architecture, tokenisation (BPE), fine-tuning, why RoBERTa-base]]

## Shortcut Learning and Trigger Words



### Shortcut Learning in Deep Networks



Shortcut learning is a phenomenon in which deep learning architectures optimise empirical risk by picking up superficial, localised data artefacts instead of learning holistic contextual structure [@geirhos2020shortcut]. Models learn to minimise empirical risk by exploiting artefacts present in the dataset, rather than learning to reason. While traditional evaluation metrics mask these vulnerabilities under high test-set accuracy, such models remain fragile when deployed out of distribution or in adversarial settings [@zhang2020adversarial].



### Lexical Shortcuts in Misinformation Detection



In textual classification, shortcut learning frequently manifests as an overwhelming reliance on sensationalised keywords, emotional adjectives or distinct stylistic formatting present in tabloid datasets, rather than on factual consistency or evidential claims [@ma2024graphing; @wan2025truth]. These models frequently overfit to task-irrelevant lexical signals known as "trigger words", such as hyperpartisan adjectives, particular punctuation or emotive formatting, rather than assessing the factual or contextual coherence of a claim [@ma2024graphing; @wan2025truth].

Two studies from Table 1 (Section 2.5) are relevant here. Wan et al. [@wan2025truth] found that models depend strongly on task-irrelevant lexical cues (for example emotional adjectives) to classify truthfulness; the remaining gap is that they do not evaluate how shortcut density changes across document lengths. Ma et al. [@ma2024graphing] showed that classifiers use sensationalised formatting and specific localised words rather than factual consistency; their work is mitigation-focused and lacks a comparative tracking baseline (MoRF versus LeRF and Random).

## Post-hoc Explainability for Text Classifiers



### Perturbation-Based Methods: LIME and SHAP



Post-hoc explainability methods for these classifiers fall into two broad families. Perturbation-based methods, LIME [@ribeiro2016why] and SHAP [@lundberg2017unified], treat the model as a black box and sample local input perturbations to fit an interpretable surrogate or to estimate Shapley-value feature contributions. They require no access to model internals but can be computationally expensive and offer no formal guarantee of sensitivity to relevant features.

### Gradient-Based Methods and Integrated Gradients



Gradient-based methods instead exploit direct access to the network. Sensitivity analysis (vanilla gradients) is cheap but can assign zero importance to features where the prediction function saturates. Integrated Gradients resolves this by integrating gradients along a path from a neutral baseline to the input, satisfying the Sensitivity and Completeness axioms that sampling-based and vanilla-gradient methods do not [@sundararajan2017axiomatic]. This deterministic, axiom-satisfying property is why this thesis adopts IG rather than LIME, SHAP or raw saliency for token-level attribution.

$$
\mathrm{IntegratedGrads}_i(x) = (x_i - x'_i) \times \int_{0}^{1} \frac{\partial F\left(x' + \alpha (x - x')\right)}{\partial x_i}\, d\alpha \tag{1}
$$

Integrated Gradients is computed along a continuous path from a baseline tensor to the input embedding; the proposal plans an all-zero baseline and a Riemann-sum approximation with $m = 50$ steps, implemented with the Captum library [@kokhlikyan2020captum; @sundararajan2017axiomatic]. Unlike sampling-based alternatives, IG avoids non-deterministic variance across runs and resolves gradient saturation in deep networks.

Two limitations are examined later (RQ3). First, the all-zero input embedding vector does not correspond to a valid textual input, which can introduce attribution artefacts. Second, because sub-word tokenisation (Byte-Pair Encoding) fragments individual words, word-level importance has to be reconstructed; summing the signed attribution scalars of all constituent sub-tokens preserves directional gradient information and gives a word-level ranking, but alternative aggregations across embedding dimensions may distort ranks through gradient cancellation [@kokhlikyan2020captum].



### Evaluating Attributions: MoRF and LeRF



Without a LeRF control track and a random-permutation baseline alongside the primary MoRF track, it is impossible to tell whether a model's prediction failure stems from removing genuine trigger words or simply from degrading the sentence's structure and grammar [@samek2017evaluating]. The three tracks used in this thesis are:

- **Most-Relevant-First (MoRF):** mutates words in descending order of attribution importance, to probe high-weight trigger markers directly.
- **Least-Relevant-First (LeRF):** mutates words in ascending order of attribution importance, as a low-attribution control baseline.
- **Random control:** mutates uniformly random words across 10 random seeds per sample, to establish baseline variance from general editing.



## Text Perturbation and Semantic Preservation



Existing frameworks treat token substitution almost exclusively as an adversarial evasion tool, designed to maximise target degradation or to optimise label-flipping vectors, rather than as a controlled white-box diagnostic that isolates and quantifies shortcut learning [@hossam2020explain2attack; @zhang2020adversarial]. Jin et al. [@jin2020bert] proved that highly accurate models such as BERT are fragile to adversarial synonym replacements of highly weighted keywords, but they treat token substitution strictly as a black-box attack technique and not as an auditable one. Hossam et al. [@hossam2020explain2attack] showed that models are highly vulnerable to label flipping when a small subset of high-influence keywords is targeted and replaced with synonyms; the approach optimises for black-box evasion rather than white-box diagnostics and lacks cross-register profiling and control baselines.

The mutation approach planned in this thesis uses a pre-trained BERT-base-uncased masked language model [@garg2020bae] to propose replacements. Candidate replacements are sampled from the top 50 MLM predictions and must pass two filters. A part-of-speech (POS) filter discards candidates that do not match the grammatical category of the original word [@garg2020bae; @jin2020bert]. A sentence-transformer encoder then measures embedding similarity between the original and mutated text and rejects substitutions below a cosine-similarity threshold of 0.85.



## Regulatory Context: Article 15 of the EU AI Act





[[Draft: what Art. 15 requires on accuracy, robustness and cybersecurity, and how a robustness audit relates. Keep to what the text says; cite the regulation text.]]

## Synthesis and Research Gap



Table 1 summarises the key studies that have diagnosed shortcut and trigger-word vulnerabilities in classification models.


| Study                                     | Domain and task                           | Identified vulnerability (trigger words / shortcuts)                                                                                  | Remaining research gap                                                                                                   |
| ----------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Geirhos et al. [@geirhos2020shortcut]     | General DNNs (vision and NLP)             | Models learn to minimise empirical risk by exploiting artefacts in the dataset, rather than learning to reason.                       | Wide theory base, but no automated white-box diagnostics pipeline for NLP.                                               |
| Jin et al. [@jin2020bert]                 | Text classification and entailment        | Highly accurate models (such as BERT) are fragile to adversarial synonym replacements of highly weighted keywords.                    | Considers token substitution strictly as a black-box attack technique, not as an auditable one.                          |
| Hossam et al. [@hossam2020explain2attack] | Text classification (adversarial attacks) | Models are highly vulnerable to label flipping when a small subset of high-influence keywords is targeted and replaced with synonyms. | Optimises for black-box evasion rather than white-box diagnostics; lacks cross-register profiling and control baselines. |
| Wan et al. [@wan2025truth]                | Misinformation detection                  | Models depend strongly on task-irrelevant lexical cues (for example emotional adjectives) to classify truthfulness.                   | Does not evaluate how shortcut density changes across varying document lengths.                                          |
| Ma et al. [@ma2024graphing]               | Multimodal fake news detection            | Classifiers leverage sensationalised formatting and specific localised words rather than evaluating factual consistency.              | Mitigation-focused; lacks a comparative tracking baseline (MoRF versus LeRF and Random).                                 |


: Comparative analysis of existing literature addressing shortcut learning and trigger-word vulnerabilities in classification models. {#tbl:related-work}

Existing frameworks treat token substitution almost exclusively as an adversarial evasion tool rather than a controlled, white-box diagnostic, and this limitation exposes three gaps in the current literature [@hossam2020explain2attack; @zhang2020adversarial]:

1. **Absence of intrinsic deterministic attribution auditing.** State-of-the-art adversarial frameworks often rely on auxiliary black-box optimisation tools, such as separate selector networks or heuristic randomised search strategies, to locate influential keywords for perturbation [@hossam2020explain2attack]. This is efficient for constructing evasion attacks, but it provides no inherent, deterministic, precise white-box audit of what the inner layers of the architecture use.
2. **Lack of cross-register and scale profiling.** Auditing systems as currently defined have not examined how trigger-word weaknesses operate under different linguistic styles and text corpora. The proposal states that no existing research has compared how the spatial clustering of lexical shortcuts behaves in short-form and long-form text [@verma2021welfake; @wang2017liar].
3. **Deficiency in structural control baselines.** Existing literature generally measures classifier vulnerability only through targeted deletions or substitutions. Without a LeRF control track and a random-permutation baseline alongside the primary MoRF track, it is impossible to tell whether a prediction failure stems from removing genuine trigger words or from degrading the sentence's structure and grammar [@samek2017evaluating].

