# Literature Matrix: Medical Hallucination Detection with RAG, NLI, Explainability, and Severity

Date prepared: 2026-05-27

Note: Direct Scopus querying was not available in this workspace. This matrix uses primary publisher, ACL, arXiv, OpenReview, PubMed/PMC, PMLR, Nature, and Hugging Face records. Before final thesis/paper submission, verify Scopus indexing/export metadata for each selected paper through institutional Scopus access.

## Working Research Direction

Develop a context-aware, explainable, fine-grained hallucination detection framework for medical NLP that unifies retrieval-augmented evidence retrieval, natural language inference verification, claim-level decomposition, hallucination type classification, and clinical severity scoring.

## Core Literature Corpus

| No. | Paper | Year | Venue/source | Main contribution | Limitation relevant to our gap |
|---:|---|---:|---|---|---|
| 1 | [MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models](https://huggingface.co/papers/2502.14302) | 2025 | EMNLP / HF / arXiv | 10k PubMedQA-derived medical hallucination benchmark; reports hard-case F1 as low as 0.625 for best model. | Strong benchmark, but main task remains detection-oriented; no full claim-level RAG-NLI-explainability-severity framework. |
| 2 | [UTAustin-AIHealth/MedHallu dataset](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | 2025 | Hugging Face dataset | Provides `pqa_labeled` and `pqa_artificial`, difficulty levels, and hallucination category fields. | Should be external test set; categories exist but are not sufficient alone for clinical severity and evidence-grounded explanations. |
| 3 | [From RAG to Reality: Coarse-Grained Hallucination Detection via NLI Fine-Tuning](https://aclanthology.org/2025.sdp-1.34.pdf) | 2025 | ACL SDP | Frames scientific hallucination detection as entailment/contradiction/unverifiable using DeBERTa-V3-large. | Scientific domain, coarse 3-way NLI labels, not medical-specific, no clinical severity. |
| 4 | [Overview of the SciHal25 Shared Task on Hallucination Detection for Scientific Content](https://aclanthology.org/2025.sdp-1.29/) | 2025 | ACL SDP | Scientific claim hallucination task with coarse and fine-grained labels over reference abstracts. | Scientific assistant setting; not clinical/biomedical safety oriented. |
| 5 | [Natural Language Inference Fine-tuning for Scientific Hallucination Detection](https://aclanthology.org/2025.sdp-1.33.pdf) | 2025 | ACL SDP | Uses NLI-style fine-tuning for scientific hallucination detection. | Useful NLI baseline but not medical-specific and no severity/explainability module. |
| 6 | [RT4CHART: Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG](https://arxiv.org/abs/2603.27752) | 2026 | arXiv | Decomposes RAG outputs into claims and verifies them hierarchically as entailed, contradicted, or baseless. | General RAG, not medical; no medical type taxonomy or clinical risk scoring. |
| 7 | [ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7daf60e805e596c3bd1e843e72ea5560-Abstract-Conference.html) | 2025 | ICLR | Uses mechanistic interpretability to analyze parametric knowledge vs retrieved context in RAG hallucination. | Internal/mechanistic focus; not end-user clinical explainability; not medical. |
| 8 | [HALT-RAG: A Task-Adaptable Framework for Hallucination Detection with Calibrated NLI Ensembles and Abstention](https://arxiv.org/abs/2509.07475) | 2025 | arXiv | NLI ensemble features plus calibrated classifier and abstention for RAG hallucination. | General benchmark focus; no medical claim taxonomy or severity. |
| 9 | [Grounded in Context: Retrieval-Based Method for Hallucination Detection](https://arxiv.org/abs/2504.15771) | 2025 | arXiv | Combines retrieval and NLI models for production-scale hallucination detection. | Production RAG, not medical; no clinical risk weighting. |
| 10 | [LettuceDetect: A Hallucination Detection Framework for RAG Applications](https://arxiv.org/abs/2502.17125) | 2025 | arXiv | ModernBERT token classification for RAG hallucination detection on RAGTruth. | Token/span detection, but no medical severity or clinical explanations. |
| 11 | [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://huggingface.co/papers/2401.00396) | 2024 | arXiv / HF | Word-level hallucination annotations for RAG outputs. | General RAG corpus; not medical-specific. |
| 12 | [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | NeurIPS Datasets & Benchmarks | Fine-grained diagnostic metrics for RAG retrieval and generation. | Evaluation framework, not a medical hallucination detector. |
| 13 | [SemEval-2025 Task 3: Mu-SHROOM](https://arxiv.org/abs/2504.11975) | 2025 | SemEval / arXiv | Multilingual hallucination and overgeneration span-labeling shared task. | Multilingual general domain; not medical and no clinical risk. |
| 14 | [HalluSearch at SemEval-2025 Task 3](https://arxiv.org/abs/2504.10168) | 2025 | SemEval / arXiv | Search-enhanced RAG pipeline for hallucination span detection. | Span localization useful, but not biomedical and not severity-aware. |
| 15 | [MSA at SemEval-2025 Task 3](https://arxiv.org/abs/2505.20880) | 2025 | SemEval / arXiv | Weak labeling and LLM ensemble verification for multilingual hallucination spans. | Uses LLM adjudication; not medical and no NLI clinical pipeline. |
| 16 | [HausaNLP at SemEval-2025 Task 3](https://arxiv.org/abs/2503.19650) | 2025 | SemEval / arXiv | Fine-grained model-aware hallucination span detection. | General multilingual task; not clinical. |
| 17 | [MedHallBench: A New Benchmark for Assessing Hallucination in Medical Large Language Models](https://proceedings.mlr.press/v281/zuo25b.html) | 2025 | PMLR / AAAI Bridge | Medical hallucination benchmark for medical LLMs. | Benchmark-oriented; does not provide a unified RAG-NLI explainable detector. |
| 18 | [MedVH: Toward Systematic Evaluation of Hallucination for Large Vision Language Models in the Medical Context](https://pubmed.ncbi.nlm.nih.gov/40843006/) | 2025 | Advanced Intelligent Systems / PubMed | Medical vision-language hallucination evaluation. | Multimodal/VLM focus, not text-only clinical claim verification. |
| 19 | [MedHallTune: An Instruction-Tuning Benchmark for Mitigating Medical Hallucination in Vision-Language Models](https://arxiv.org/abs/2502.20780) | 2025 | arXiv | Instruction-tuning benchmark for medical VLM hallucination mitigation. | VLM mitigation, not explainable text hallucination detection. |
| 20 | [MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models](https://arxiv.org/abs/2503.02157) | 2025 | arXiv | Medical LVLM hallucination benchmark with cause categories. | Visual focus; no text RAG-NLI severity framework. |
| 21 | [Med-StepBench: Hierarchical Reasoning for Hallucination Detection in Medical VLMs](https://arxiv.org/abs/2605.10002) | 2026 | arXiv | Step-wise hallucination detection in 3D oncological PET/CT. | Imaging-specific; useful for hierarchical inspiration only. |
| 22 | [A Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation](https://pubmed.ncbi.nlm.nih.gov/40360677/) | 2025 | npj Digital Medicine / PubMed | Clinical summarization error taxonomy, harm framework, GUI, clinician-annotated sentences. | Strong clinical safety reference, but not RAG-NLI claim verification on MedHallu. |
| 23 | [Multi-model assurance analysis showing LLMs are vulnerable to adversarial hallucination attacks during clinical decision support](https://pubmed.ncbi.nlm.nih.gov/40753316/) | 2025 | Communications Medicine / PubMed | Tests clinical decision-support vulnerability under adversarial hallucination attacks. | Focuses vulnerability evaluation, not detector architecture. |
| 24 | [Large language models provide unsafe answers to patient-posed medical questions](https://pubmed.ncbi.nlm.nih.gov/41688533/) | 2025 | PubMed | Physician-led red-teaming of public chatbots on patient medical advice. | Shows clinical risk but lacks automated fine-grained detector. |
| 25 | [Large language models in healthcare: a systematic evaluation on medical Q/A datasets](https://pubmed.ncbi.nlm.nih.gov/41281608/) | 2025/2026 | PubMed | Evaluates LLMs on PubMedQA, MedQA, MedMCQA and highlights hallucination/explainability concerns. | QA evaluation, not hallucination taxonomy or severity detector. |
| 26 | [LongHealth: A Question Answering Benchmark with Long Clinical Documents](https://pubmed.ncbi.nlm.nih.gov/40726742/) | 2025 | PubMed | Long clinical-document QA benchmark for healthcare LLM evaluation. | QA benchmark, not hallucination detection pipeline. |
| 27 | [HealthBench: Evaluating Large Language Models Towards Improved Human Health](https://arxiv.org/abs/2505.08775) | 2025 | arXiv / OpenAI | 5,000 realistic health conversations with physician-created rubrics. | Broad health model evaluation; not a detector trained on hallucination types. |
| 28 | [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) | 2024 | Nature | Detects confabulations through semantic uncertainty/entropy. | General LLM method; not medical-specific or evidence-grounded. |
| 29 | [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) | 2025 | arXiv / OpenAI | Explains hallucination persistence via training/evaluation incentives rewarding guessing. | Theoretical cause analysis, not medical detection. |
| 30 | [A Comprehensive Survey of Hallucination Mitigation Techniques in Large Language Models](https://arxiv.org/abs/2401.01313) | 2024 | arXiv | Survey of hallucination mitigation techniques and taxonomy. | Broad survey; supports gap but not a medical implementation. |
| 31 | [The ethics of ChatGPT in medicine and healthcare: a systematic review on LLMs](https://www.nature.com/articles/s41746-024-01157-x) | 2024 | npj Digital Medicine | Reviews ethical implications of LLMs in healthcare. | Ethics/safety background, not technical detector. |
| 32 | [Large Language Models in Healthcare and Medical Domain: A Review](https://www.mdpi.com/2227-9709/11/3/57) | 2024 | Informatics | Reviews healthcare LLM applications and concerns. | Broad review; lacks detector design. |

## Synthesized Research Gaps

### Gap 1: Medical hallucination detection is still not fully claim-level and type-aware.

MedHallu provides a major medical benchmark and includes hallucination category fields, while SciHal and Mu-SHROOM show that the wider NLP community is moving toward claim-level and span-level hallucination detection. However, existing medical text hallucination systems do not yet provide a complete pipeline that decomposes medical answers into atomic claims, assigns fine-grained hallucination types to each claim, and evaluates those claims against retrieved biomedical evidence.

### Gap 2: RAG and NLI are combined in general systems, but not as a clinically grounded medical detector.

General-domain systems such as HALT-RAG, Grounded in Context, LettuceDetect, From RAG to Reality, and RT4CHART show that retrieval, NLI, token/span detection, and claim verification are promising. The missing step is adapting this into a medical-domain pipeline using biomedical evidence sources, medical entailment models, MedHallu/PubMedQA-style evaluation, and clinical risk logic.

### Gap 3: Explainability is usually technical or absent, not clinician-facing.

ReDeEP offers mechanistic interpretability, and some RAG tools return spans or labels. Clinical users need evidence snippets, highlighted contradicted tokens, type labels, and a readable reason for why a claim is unsafe. Existing medical hallucination benchmarks mostly report labels, rates, or scores rather than usable end-user explanations.

### Gap 4: Clinical severity is under-modeled.

Clinical hallucination risk is not uniform. A wrong drug, dose, diagnosis, contraindication, or causal mechanism can be much more dangerous than an irrelevant contextual statement. Clinical safety work recognizes harm, but current hallucination detectors rarely combine hallucination type, contradiction strength, and clinical risk tier into a formal severity score.

### Gap 5: Benchmark strength exists, but cross-benchmark validation is thin.

MedHallu is the key external benchmark, while PubMedQA, HealthBench, clinical summarization datasets, and RAGTruth-style datasets offer useful complementary testing. A strong project should evaluate internally on annotated training/validation data and externally on MedHallu, especially the hard split.

## Final Research Gap Statement

Recent work has advanced medical hallucination benchmarks, scientific NLI-based hallucination detection, RAG hallucination diagnostics, and clinical safety evaluation separately. However, no existing work provides a unified medical text hallucination detection framework that performs claim-level decomposition, retrieves biomedical evidence, verifies each claim using NLI, assigns fine-grained hallucination types, explains the decision through evidence and token-level attribution, and maps the result to clinical severity. This gap is important because current models still struggle on hard medical hallucinations, and medical deployment requires not only detection accuracy but also interpretable evidence and risk-aware prioritization.

## Proposed Objective

To develop and evaluate an explainable, context-aware, claim-level medical hallucination detection framework that integrates biomedical RAG, NLI-based verification, fine-grained hallucination type classification, and clinical severity scoring.

## Proposed Sub-objectives

1. Build a claim decomposition module for medical QA answers and generated clinical text.
2. Retrieve biomedical evidence from PubMedQA/PubMed abstracts or another curated medical knowledge base.
3. Verify each atomic claim using an NLI cross-encoder with labels such as supported, contradicted, and unverifiable.
4. Classify hallucination type using a fine-grained taxonomy aligned with MedHallu categories and clinically meaningful error classes.
5. Generate explanations using retrieved evidence spans, highlighted contradicted tokens, and feature attribution.
6. Score clinical severity using hallucination type, contradiction confidence, and clinical risk tier.
7. Evaluate against MedHallu, especially hard cases, and compare with binary classifiers, NLI-only baselines, and RAG-only baselines.

## Recommended Datasets

| Dataset | Use in project |
|---|---|
| MedHallu | Main external benchmark and target test set. |
| PubMedQA | Evidence source and possible training/evaluation base. |
| RAGTruth / RAGTruth++ | General RAG hallucination comparison or pretraining/ablation reference. |
| SciHal25 | Scientific NLI claim verification reference. |
| Mu-SHROOM | Span-level hallucination detection reference. |
| HealthBench | Optional clinical safety/rubric inspiration, not core detector training. |
| Clinical summarization safety dataset from Asgari et al. | Inspiration for harm/severity taxonomy. |

## Suggested Baselines

1. Binary classifier on question-answer pairs.
2. NLI-only verifier without retrieval.
3. RAG retrieval plus similarity scoring.
4. RAG plus NLI without type classification.
5. Full proposed system: claim decomposition + RAG + NLI + type + explanation + severity.

## Strongest Expected Contribution

The strongest contribution is not simply another hallucination detector. It is a clinically aware, evidence-grounded, interpretable hallucination verification framework that identifies what is wrong, why it is wrong, where the evidence is, what type of hallucination it is, and how clinically severe it may be.
