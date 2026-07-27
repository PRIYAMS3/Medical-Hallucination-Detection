# Implementation Roadmap

## Phase 1: Dataset and Baseline Setup

Goal: create a clean reproducible base.

Tasks:

1. Create Python package structure.
2. Load MedHallu dataset from Hugging Face.
3. Inspect fields, labels, categories, and difficulty splits.
4. Build binary baseline using question + answer features.
5. Report accuracy, precision, recall, F1, and hard-case F1.

Output:

- `data/` loading scripts.
- Baseline results table.
- Dataset inspection report.

## Phase 2: Claim Decomposition

Goal: move from answer-level detection to claim-level detection.

Tasks:

1. Implement rule-based sentence and clause splitting.
2. Add optional LLM-assisted decomposition later.
3. Store claim-level records with parent sample IDs.
4. Evaluate whether claim-level aggregation improves answer-level detection.

Output:

- Claim decomposition module.
- Claim-level dataset file.

## Phase 3: Retrieval Module

Goal: retrieve evidence for each claim.

Tasks:

1. Build BM25 retrieval over PubMedQA/MedHallu context.
2. Add dense retrieval with biomedical embeddings.
3. Add hybrid retrieval.
4. Evaluate retrieval hit rate and Recall@k.

Output:

- Evidence index.
- Retrieval evaluation table.

## Phase 4: NLI Verification

Goal: verify claims against retrieved evidence.

Tasks:

1. Pair each claim with top-k evidence.
2. Run NLI model.
3. Aggregate passage-level NLI scores into claim-level labels.
4. Compare NLI-only vs RAG + NLI.

Output:

- Supported/contradicted/unverifiable labels.
- NLI baseline and RAG-NLI results.

## Phase 5: Type Classification

Goal: classify hallucination type.

Tasks:

1. Map MedHallu category field to final taxonomy.
2. Add clinically meaningful categories if manually annotated.
3. Train a multi-class or multi-label type classifier.
4. Evaluate macro-F1 and per-type performance.

Output:

- Type taxonomy.
- Type classifier.
- Confusion matrix.

## Phase 6: Explainability

Goal: make the output interpretable.

Tasks:

1. Show retrieved evidence spans.
2. Highlight claim tokens contributing to contradiction/type prediction.
3. Add SHAP or gradient-based attribution for model features.
4. Generate a per-sample explanation report.

Output:

- Explanation JSON.
- Optional Streamlit/FastAPI demo.

## Phase 7: Clinical Severity Scoring

Goal: prioritize clinically dangerous hallucinations.

Tasks:

1. Define type weights.
2. Define clinical risk tiers.
3. Calculate severity score.
4. Evaluate high-risk recall and manual severity agreement.

Output:

- Severity scoring module.
- Severity analysis table.

## Phase 8: Final Evaluation and Paper Package

Goal: produce final research results.

Tasks:

1. Run all baselines.
2. Run full framework.
3. Compare performance on full MedHallu and hard split.
4. Create ablation study.
5. Write results, discussion, limitations, and future work.

Output:

- Final result tables.
- Architecture diagram.
- Paper draft.
- Reproducible code package.

## Minimal Viable Research Prototype

If time is limited, build only:

1. MedHallu loader.
2. Binary baseline.
3. Claim decomposition.
4. BM25 retrieval.
5. NLI verification.
6. Type mapping using MedHallu categories.
7. Severity formula.

This is enough to demonstrate the core contribution.

