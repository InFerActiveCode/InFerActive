# InFerActive Evaluation

This folder contains the vLLM-based evaluation scripts used for the benchmark
experiments. Dataset files, prompt CSVs, model weights, and generated results
are intentionally not committed.

## Contents

```text
evaluation/
├── run_experiment.py                  # Main vLLM tree -> continuation -> classifier pipeline
├── run_random_sampling.py             # Random sampling baseline
├── run_random_continuation.py         # Random walk continuation baseline
├── run_hybrid_random_continuation.py  # Hybrid tree/random pipeline
├── config/                            # Model and generation configs
├── models/download_model.py           # Optional model download helper
├── modules/
│   ├── tree_explorer_vllm.py          # vLLM token tree exploration
│   ├── continuation.py                # Tree continuation and greedy baseline
│   ├── classifier.py                  # HarmBench classifier runner
│   ├── classifier_preprocess.py       # Classifier input normalization
│   ├── random_sampler.py              # Random sampling utilities
│   ├── random_continuation.py         # Random continuation baseline
│   ├── hybrid_random_tree_explorer.py # Hybrid tree exploration
│   ├── hybrid_walk_continuation.py    # Hybrid continuation
│   ├── inference_engine.py            # Batched vLLM inference helper
│   ├── model_utils.py                 # Prompt templates and model loading
│   └── token_node.py                  # Token tree node structure
└── requirements.txt
```

## Setup

Install the evaluation dependencies in an environment that can run vLLM:

```bash
cd evaluation
pip install -r requirements.txt
```

Model paths are configured in `config/models.yaml`. For local release testing,
`llama3_1b` points at `../llama-3.2-1b`, which is ignored by git.

## Main Pipeline

Run the vLLM tree exploration, continuation generation, optional greedy
baseline, and optional classifier:

```bash
python run_experiment.py \
  --model-name llama3_1b \
  --config-file config/models.yaml \
  --behaviors-csv /path/to/behaviors.csv \
  --output-dir results \
  --max-count 10
```

Useful development flags:

```bash
python run_experiment.py \
  --model-name llama3_1b \
  --config-file config/models.yaml \
  --behaviors-csv /path/to/behaviors.csv \
  --output-dir results \
  --max-count 1 \
  --skip-classifier
```

Use `--skip-tree --tree-dir /path/to/trees` to reuse existing tree JSON files,
and `--resume` to continue supported generation/classification steps.

## Baselines

```bash
python run_random_sampling.py \
  --model-name llama3_1b \
  --config-file config/models.yaml \
  --behaviors-csv /path/to/behaviors.csv \
  --output-dir results/random_sampling

python run_random_continuation.py \
  --model-name llama3_1b \
  --config-file config/models.yaml \
  --behaviors-csv /path/to/behaviors.csv \
  --output-dir results/random_continuation

python run_hybrid_random_continuation.py \
  --model-name llama3_1b \
  --config-file config/models.yaml \
  --behaviors-csv /path/to/behaviors.csv \
  --output-dir results/hybrid_random
```

## Expected Inputs

The behavior CSV is external to the repository. It should include the columns
used by the experiment scripts, including behavior text and functional category
fields such as `Behavior`, `BehaviorID`, and `FunctionalCategory`.

## Outputs

Generated outputs are written under the selected output directory:

```text
results/{model_name}/
├── trees/              # JSON token trees
├── continued.csv       # Continuation results
├── greedy.csv          # Greedy baseline completions
├── classified.csv      # Classifier results for tree continuations
├── greedy_classified.csv
└── summary.json
```

The repository `.gitignore` excludes generated results and local model weights.
