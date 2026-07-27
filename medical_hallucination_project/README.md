# Explainable Medical Hallucination Detection Project

This folder contains the research proposal and implementation plan for a medical NLP project on claim-level hallucination detection.

## Project Title

Context-Aware Explainable Medical Hallucination Detection Using Retrieval-Augmented NLI, Fine-Grained Type Classification, and Clinical Severity Scoring

## Core Idea

The project detects hallucinations in medical answers by decomposing each answer into atomic claims, retrieving biomedical evidence, verifying each claim using natural language inference, classifying the hallucination type, explaining the decision with evidence and token highlights, and scoring the clinical severity of the error.

## Key Benchmark

MedHallu should be used as the main external benchmark. It contains PubMedQA-derived hallucination samples and is currently one of the most relevant medical hallucination benchmarks for this task.

## Files

- `research_proposal.md`: full research proposal.
- `methodology.md`: system architecture, modules, formulas, and evaluation plan.
- `implementation_roadmap.md`: staged build plan from baseline to full framework.

