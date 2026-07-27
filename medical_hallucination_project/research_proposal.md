# Research Proposal

## Title

Context-Aware Explainable Medical Hallucination Detection Using Retrieval-Augmented NLI, Fine-Grained Type Classification, and Clinical Severity Scoring

## Background

Large language models are increasingly used for medical question answering, clinical summarization, biomedical literature assistance, and patient-facing health information. However, these models can produce hallucinations: fluent statements that are unsupported, contradicted by evidence, or clinically unsafe. In medicine, hallucinations are more serious than ordinary factual errors because they may affect diagnosis, treatment, medication advice, contraindications, patient safety, or clinical decision-making.

Recent literature shows rapid progress but also fragmentation. MedHallu introduced a medical hallucination benchmark derived from PubMedQA and showed that even strong models struggle on hard medical hallucinations. Scientific hallucination detection work such as SciHal25 and From RAG to Reality has shown the usefulness of natural language inference. RAG hallucination studies such as RAGTruth, RAGChecker, HALT-RAG, ReDeEP, LettuceDetect, and RT4CHART explore retrieval grounding, span detection, mechanistic interpretability, and claim-level verification. Clinical safety studies show that hallucinations are not equally harmful; the severity depends on the medical content and potential patient impact.

Despite these advances, the literature has not yet unified claim-level medical hallucination detection, biomedical evidence retrieval, NLI verification, fine-grained hallucination type classification, clinician-facing explanation, and clinical severity scoring in one framework.

## Problem Statement

Existing medical hallucination detection methods are limited because they often focus on binary hallucination labels or benchmark-level evaluation without explaining the specific claim-level error, retrieving supporting or contradicting biomedical evidence, identifying the hallucination type, or estimating clinical risk. This creates a gap between model evaluation and real clinical usefulness.

This project addresses the problem of detecting and explaining medical hallucinations at the claim level by combining retrieval-augmented evidence grounding, NLI-based verification, fine-grained type classification, and clinical severity scoring.

## Research Gap

Recent work has advanced medical hallucination benchmarks, scientific NLI-based hallucination detection, RAG hallucination diagnostics, and clinical safety evaluation separately. However, no existing work provides a unified medical text hallucination detection framework that:

1. Decomposes medical answers into atomic claims.
2. Retrieves biomedical evidence for each claim.
3. Verifies each claim using NLI.
4. Classifies hallucination type.
5. Provides evidence-grounded explanations.
6. Scores clinical severity based on risk.

This gap matters because medical hallucination detection should not only say whether an answer is wrong. It should also say what is wrong, why it is wrong, where the evidence is, and how clinically serious the error may be.

## Aim

To develop and evaluate an explainable, context-aware, claim-level medical hallucination detection framework that integrates biomedical retrieval, natural language inference, fine-grained hallucination type classification, and clinical severity scoring.

## Objectives

1. Develop a claim decomposition module that splits medical answers into atomic verifiable claims.
2. Build a biomedical retrieval module that retrieves relevant evidence from PubMedQA, PubMed abstracts, or a curated biomedical corpus.
3. Implement an NLI-based verification module that classifies each claim as supported, contradicted, or unverifiable.
4. Design a fine-grained hallucination taxonomy for medical text and train a type classifier aligned with MedHallu categories and clinically meaningful error types.
5. Add explainability using retrieved evidence spans, token-level highlighting, and feature attribution.
6. Create a clinical severity score using hallucination type, contradiction strength, and clinical risk tier.
7. Evaluate the system on MedHallu, with special attention to hard hallucination cases, and compare against binary, NLI-only, and RAG-only baselines.

## Research Questions

1. Can claim-level decomposition improve medical hallucination detection compared with answer-level binary classification?
2. Does combining biomedical retrieval with NLI verification improve detection performance on hard MedHallu samples?
3. Can fine-grained hallucination type classification provide more useful error analysis than binary labels alone?
4. Can evidence spans and token-level attribution make hallucination detection more interpretable for medical users?
5. Can a severity score better prioritize clinically risky hallucinations than confidence scores alone?

## Proposed Contribution

The project contributes a unified medical hallucination detection framework that combines:

- Claim-level medical text verification.
- Biomedical RAG evidence retrieval.
- NLI-based entailment, contradiction, and unverifiable classification.
- Fine-grained hallucination type detection.
- Explanation through evidence display and token highlighting.
- Clinical severity scoring.
- External benchmark evaluation on MedHallu.

## Expected Outcome

The expected outcome is a working research prototype and experimental report showing whether the proposed hybrid framework improves hallucination detection on medical text, especially hard MedHallu examples, while also producing interpretable type and severity outputs.

