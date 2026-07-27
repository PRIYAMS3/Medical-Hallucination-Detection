# Running the Implementation

All implementation commands should be run from this folder:

```powershell
cd "C:\Users\PRIYAMVADA NAMBIAR\OneDrive - Amrita Vishwa Vidyapeetham\Documents\New project\medical_hallucination_project"
```

Set the Python path before running project modules:

```powershell
$env:PYTHONPATH="src"
```

If Hugging Face download commands fail because of a broken proxy, clear proxy variables for the current PowerShell session:

```powershell
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""
$env:GIT_HTTP_PROXY=""
$env:GIT_HTTPS_PROXY=""
```

## Step 1: Install Minimum Dependency

```powershell
python -m pip install datasets
python -m pip install "aiohttp==3.9.5"
```

## Step 2: Download MedHallu

The direct downloader avoids Hugging Face cache/symlink issues on Windows + OneDrive.

```powershell
python -B -m medhallu_pipeline.download_medhallu_direct --split pqa_labeled
```

Expected local dataset file:

```text
data\medhallu\pqa_labeled.parquet.part
```

The loader can read either `.parquet` or `.parquet.part`.

## Step 3: Inspect Dataset

```powershell
python -B -m medhallu_pipeline.inspect_dataset --split pqa_labeled --out outputs\dataset_inspection_pqa_labeled.json
```

Observed dataset structure:

- Rows: 1000
- Columns:
  - `Question`
  - `Knowledge`
  - `Ground Truth`
  - `Difficulty Level`
  - `Hallucinated Answer`
  - `Category of Hallucination`

## Step 4: Run Binary Baseline

```powershell
python -B -m medhallu_pipeline.baseline_binary --split pqa_labeled --out outputs\binary_baseline_pqa_labeled_report.json
```

Current baseline:

- Model: TF-IDF + Logistic Regression
- Examples: 2000 answer examples from 1000 MedHallu rows
- Ground Truth answer = `not_hallucinated`
- Hallucinated Answer = `hallucinated`
- Initial accuracy: about 0.17

This weak result is expected for a naive lexical baseline and motivates the stronger claim-level RAG + NLI approach.

## Next Implementation Step

## Step 5: Build Claim-Level Dataset

```powershell
python -B -m medhallu_pipeline.build_claim_dataset --split pqa_labeled --out outputs\claim_level_pqa_labeled.csv
```

Current output:

```text
4652 claim records
```

## Step 6: Run Retrieval

```powershell
python -B -m medhallu_pipeline.run_retrieval --claims outputs\claim_level_pqa_labeled.csv --out outputs\retrieval_pqa_labeled_top5.csv --top-k 5
```

Current output:

```text
23260 claim-evidence rows
```

## Step 7: Retrieval Baseline

```powershell
python -B -m medhallu_pipeline.retrieval_baseline --retrieval outputs\retrieval_pqa_labeled_top5.csv --out outputs\retrieval_baseline_report.json
```

Current result:

```text
Accuracy: 0.412
Macro-F1: 0.337
Hallucinated F1: 0.561
```

## Step 8: Heuristic RAG + NLI

```powershell
python -B -m medhallu_pipeline.run_heuristic_nli --retrieval outputs\retrieval_pqa_labeled_top5.csv --out outputs\heuristic_nli_pqa_labeled_top1.csv --max-rank 1
python -B -m medhallu_pipeline.evaluate_nli_outputs --nli outputs\heuristic_nli_pqa_labeled_top1.csv --out outputs\heuristic_nli_eval_top1_report.json
```

Current result:

```text
Accuracy: 0.443
Macro-F1: 0.418
Hallucinated F1: 0.540
Hard hallucinated F1: 0.555
```

## Step 9: Type Classification

```powershell
python -B -m medhallu_pipeline.type_classifier --claims outputs\claim_level_pqa_labeled.csv --out outputs\type_classifier_report.json
```

Current result:

```text
Accuracy: 0.840
Macro-F1: 0.803
Weighted-F1: 0.810
```

## Next Implementation Step

The next step is transformer-based NLI verification:

1. Run a real NLI cross-encoder on claim-evidence pairs.
2. Compare it against the heuristic NLI baseline.
3. Use contradiction probability as `contradiction_strength` for severity scoring.

Transformer NLI runner:

```powershell
python -B -m medhallu_pipeline.run_transformer_nli --retrieval outputs\retrieval_pqa_labeled_top5.csv --out outputs\transformer_nli_smoke20.csv --limit 20 --batch-size 4
```

Current note:

The runner is implemented, but model download failed in the current OneDrive folder because Hugging Face could not safely move cache files. If this happens, move the whole project to a normal local path such as `C:\medhallu_project` or manually download the model.
