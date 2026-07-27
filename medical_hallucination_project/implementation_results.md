# Implementation Results Log

Date: 2026-05-28

## Dataset Loaded

Dataset: `UTAustin-AIHealth/MedHallu`

Config used: `pqa_labeled`

Local file used by loader:

```text
data/medhallu/pqa_labeled.parquet.part
```

Rows: 1000

Columns:

- `Question`
- `Knowledge`
- `Ground Truth`
- `Difficulty Level`
- `Hallucinated Answer`
- `Category of Hallucination`

Difficulty distribution:

| Difficulty | Rows |
|---|---:|
| hard | 408 |
| medium | 318 |
| easy | 274 |

Hallucination category distribution:

| Category | Rows |
|---|---:|
| Misinterpretation of #Question# | 752 |
| Incomplete Information | 212 |
| Mechanism and Pathway Misattribution | 33 |
| Methodological and Evidence Fabrication | 3 |

## Binary Answer-Level Baseline

Input construction:

- `Ground Truth` answer -> `not_hallucinated`
- `Hallucinated Answer` -> `hallucinated`

Total examples: 2000

Model:

- TF-IDF vectorizer
- Logistic regression classifier

Held-out test result:

| Metric | Value |
|---|---:|
| Accuracy | 0.170 |
| Macro-F1 | 0.170 |
| Hallucinated F1 | 0.153 |
| Not hallucinated F1 | 0.186 |

Interpretation:

The answer-level lexical classifier is very weak. This supports the need for evidence-grounded claim-level verification rather than simple surface-form classification.

## Claim-Level Dataset

Generated file:

```text
outputs/claim_level_pqa_labeled.csv
```

Rows: 4652 claim records

Claim label distribution:

| Label | Claims |
|---|---:|
| not_hallucinated | 2846 |
| hallucinated | 1806 |

Difficulty distribution at claim level:

| Difficulty | Claims |
|---|---:|
| hard | 1862 |
| medium | 1483 |
| easy | 1307 |

Hallucinated claim category distribution:

| Category | Claims |
|---|---:|
| Misinterpretation of #Question# | 1327 |
| Incomplete Information | 409 |
| Mechanism and Pathway Misattribution | 60 |
| Methodological and Evidence Fabrication | 10 |

## Retrieval Baseline

Generated file:

```text
outputs/retrieval_pqa_labeled_top5.csv
```

Retrieval setting:

- Retriever: TF-IDF lexical retrieval with small medical synonym expansion
- Corpus: unique MedHallu `Knowledge` passages
- Top-k: 5

Rows: 23260 claim-evidence pairs

Top-1 evidence score:

| Statistic | Value |
|---|---:|
| Mean top-1 score | 0.322 |
| Median top-1 score | 0.300 |
| Mean hallucinated top-1 score | 0.319 |
| Mean not hallucinated top-1 score | 0.324 |

Retrieval-score threshold baseline:

| Metric | Value |
|---|---:|
| Accuracy | 0.412 |
| Macro-F1 | 0.337 |
| Hallucinated F1 | 0.561 |
| Not hallucinated F1 | 0.113 |
| Best threshold | 0.61 |

Interpretation:

Retrieval score alone is not enough because hallucinated and non-hallucinated claims have very similar top evidence similarity. However, retrieval provides relevant context that can be passed to an NLI verifier. This motivates the next module: RAG + NLI.

## Next Step

## Heuristic RAG + NLI Baseline

Generated file:

```text
outputs/heuristic_nli_pqa_labeled_top1.csv
```

Setting:

- Evidence: top-1 retrieved evidence passage per claim
- Verifier: transparent heuristic NLI-style verifier
- Labels: `supported`, `contradicted`, `unverifiable`
- Hallucination prediction rule: `contradicted` or `unverifiable` = hallucinated

Label distribution:

| NLI label | Claims |
|---|---:|
| contradicted | 1972 |
| unverifiable | 1855 |
| supported | 825 |

Overall evaluation:

| Metric | Value |
|---|---:|
| Accuracy | 0.443 |
| Macro-F1 | 0.418 |
| Hallucinated F1 | 0.540 |
| Not hallucinated F1 | 0.295 |

Hard split evaluation:

| Metric | Value |
|---|---:|
| Hard accuracy | 0.454 |
| Hard macro-F1 | 0.425 |
| Hard hallucinated F1 | 0.555 |

Severity distribution after MedHallu category weighting:

| Severity | Claims |
|---|---:|
| low | 4252 |
| moderate | 347 |
| high | 53 |

Interpretation:

The heuristic verifier improves over pure retrieval for claim-level hallucinated F1 and creates the full output structure needed for explanation and severity. However, it still overpredicts hallucination because many correct claims are marked contradicted or unverifiable. This motivates a transformer NLI cross-encoder.

## Fine-Grained Type Classification Baseline

Generated file:

```text
outputs/type_classifier_report.json
```

Input:

- Hallucinated claims only
- Text features: question + claim + knowledge
- Labels: MedHallu `Category of Hallucination`

Model:

- TF-IDF vectorizer
- Class-weighted logistic regression

Results:

| Metric | Value |
|---|---:|
| Accuracy | 0.840 |
| Macro-F1 | 0.803 |
| Weighted-F1 | 0.810 |

Per-category F1:

| Category | F1 |
|---|---:|
| Misinterpretation of #Question# | 0.902 |
| Incomplete Information | 0.509 |
| Mechanism and Pathway Misattribution | 0.800 |
| Methodological and Evidence Fabrication | 1.000 |

Important caution:

`Methodological and Evidence Fabrication` has very few examples, so its perfect score is not reliable. It should be reported as a low-support class.

## Next Step

Implement transformer-based NLI verification over claim-evidence pairs:

1. Install `transformers`.
2. Run a lightweight NLI model over top-1 or top-3 evidence.
3. Produce supported, contradicted, and unverifiable labels.
4. Compare RAG + NLI against binary and retrieval-only baselines.

Current status:

- Transformer NLI runner has been implemented in `src/medhallu_pipeline/run_transformer_nli.py`.
- A 20-claim smoke test was attempted with `typeform/distilbert-base-uncased-mnli`.
- The script reached model download/loading, but Hugging Face model caching failed on Windows/OneDrive due to cache file rename/move errors.
- This is an environment/cache issue, not a pipeline design issue.

Workaround options:

1. Run the project from a non-OneDrive folder such as `C:\medhallu_project`.
2. Enable Windows Developer Mode or run Python as administrator so Hugging Face symlinks work.
3. Manually download the NLI model into a local folder and pass that folder to `--model`.
4. Continue development using the heuristic NLI baseline until the cache issue is resolved.

Update:

Using `C:\medhallu_hf_cache` solved the OneDrive cache issue. A 20-claim smoke test passed, and a 500-claim transformer NLI chunk was completed.

Transformer NLI 500-claim sample:

| Metric | Value |
|---|---:|
| Accuracy | 0.426 |
| Macro-F1 | 0.418 |
| Hallucinated F1 | 0.487 |
| Hard hallucinated F1 | 0.492 |

Interpretation:

The generic DistilBERT MNLI model is not better than the heuristic verifier on this medical claim-verification task. This supports the need for either biomedical NLI adaptation or supervised calibration on MedHallu-style claim-evidence pairs.

## Supervised RAG Verifier

Generated files:

```text
outputs/supervised_rag_verifier_report.json
outputs/supervised_rag_verifier_tuned_report.json
```

Setting:

- Input: claim + top-1 retrieved evidence
- Model: TF-IDF + class-weighted logistic regression
- Split: grouped by original MedHallu row to reduce leakage

Default classifier threshold:

| Metric | Value |
|---|---:|
| Accuracy | 0.547 |
| Macro-F1 | 0.529 |
| Hallucinated F1 | 0.435 |
| Hard hallucinated F1 | 0.447 |

Hallucinated-F1 tuned threshold:

| Metric | Value |
|---|---:|
| Accuracy | 0.380 |
| Macro-F1 | 0.291 |
| Hallucinated F1 | 0.543 |
| Hard hallucinated F1 | 0.565 |

Interpretation:

The supervised verifier improves overall accuracy and macro-F1 at the default threshold. Threshold tuning improves hallucinated recall and hard hallucinated F1, but harms non-hallucinated detection. The best hard hallucinated F1 so far is 0.565, still below the target benchmark of 0.625.

## Current Baseline Comparison

Generated file:

```text
outputs/results_comparison.csv
```

| System | Accuracy | Macro-F1 | Hallucinated F1 | Hard hallucinated F1 |
|---|---:|---:|---:|---:|
| Binary TF-IDF | 0.170 | 0.170 | 0.153 | NA |
| Retrieval score | 0.412 | 0.337 | 0.561 | NA |
| Heuristic RAG+NLI | 0.443 | 0.418 | 0.540 | 0.555 |
| Supervised RAG verifier | 0.547 | 0.529 | 0.435 | 0.447 |
| Supervised RAG verifier tuned | 0.380 | 0.291 | 0.543 | 0.565 |
| Transformer NLI 500-sample | 0.426 | 0.418 | 0.487 | 0.492 |

Current conclusion:

The pipeline is implemented and measurable, but model quality is not yet strong enough for a final publishable claim. The next improvement should focus on biomedical NLI/dense retrieval and better claim label handling rather than simply running a generic NLI model over more samples.
