# RePlan EMNLP 2026 Submission Code

This directory contains the core code used for RePlan, extracted from the actual experimental implementation. It is not a separate toy reimplementation. The main method code is under `replan_core/`.

## Core Files

- `replan_core/plan_generator.py`: the 2-layer autoregressive Transformer planner.
- `replan_core/pretrain_plan_generator.py`: planner pretraining with InfoNCE plus cosine alignment.
- `replan_core/replan_model_v1.py`: the final RePlan retriever with residual plan projection and relation-feedback conditioning.
- `replan_core/models.py`: the GNN layer and static/offline-plan retriever variants used for controls.
- `replan_core/base_model.py`: training, validation, test, checkpointing, and path export wrapper.
- `replan_core/train.py`: joint retriever training and test entry point.
- `replan_core/regen_path.py`: path export from a trained checkpoint.
- `replan_core/llm_rerank.py`: option-based answer-selection prompt construction and reranking interface.
- `replan_core/load_data.py`: KGQA data loader for WebQSP/CWQ-style question subgraphs.
- `replan_core/utils.py`: ranking metrics.

The paper's main method corresponds to `replan_model_v1.py` with `--projection_mode residual`. `replan_model.py` is retained only for the direct-projection branch used by `--projection_mode direct`.

## Included Sample

The directory includes a compact CWQ-format sample:

- `data/CWQ/{train,dev,test}_simple.json`
- `data/CWQ/entity_name.txt`
- `data/CWQ/relations.txt`
- `embedding/CWQ-*.npy`

The sample is remapped to a small local entity/relation ID space so the real loader and model code can be inspected without shipping the full CWQ subgraphs and Qwen embedding files. It is for smoke testing only, not for reproducing paper numbers.

## Environment

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The full training code uses PyTorch and `torch-scatter`. The original experiments were run with GPUs.

## Smoke Test

Run a lightweight check that builds the actual planner class and computes the planner pretraining loss on the included sample embeddings:

```bash
cd EMNLP2026_submission_code/replan_core
python smoke_test.py
```

Expected output:

```text
Smoke test passed.
Sample embeddings: questions=(4, 3584), plans=(4, 4, 3584)
...
```

To additionally check the real CWQ data loader, install `scipy` and run:

```bash
python check_data_loader.py
```

## Planner Pretraining

On the included sample:

```bash
cd EMNLP2026_submission_code/replan_core
python pretrain_plan_generator.py \
  --dataset CWQ \
  --emb_dir ../embedding \
  --question_emb_dir ../embedding \
  --plan_emb_dir ../embedding \
  --output_ckpt results/CWQ_plan_generator_best.pt \
  --perf_file results/CWQ_plan_generator_perf.txt \
  --epochs 1 \
  --batch_size 2
```

For full experiments, replace `../embedding` and `../data/CWQ` with the complete Qwen embedding and dataset files.

## Joint Retriever Training

Main RePlan setting:

```bash
cd EMNLP2026_submission_code/replan_core
python train.py \
  --dataset CWQ \
  --emb_dir ../embedding \
  --plan_emb_dir ../embedding \
  --projection_mode residual \
  --epochs 1 \
  --n_batch 2 \
  --gpu 0
```

This follows the real training entry point. On machines without CUDA, use the smoke test for code inspection; the full training scripts are GPU-oriented, matching the original experiment code.

## Test And Path Export

After training a checkpoint:

```bash
python train.py \
  --dataset CWQ \
  --emb_dir ../embedding \
  --plan_emb_dir ../embedding \
  --projection_mode residual \
  --test_only \
  --gen_path \
  --gpu 0
```

`llm_rerank.py` formats retrieved candidates and paths for answer selection. The submitted package does not include API keys.

## Notes

- The code in `replan_core/` is extracted from the actual RePlan experimental code.
- The included sample data is compact and only verifies the pipeline interfaces.
- No API keys, checkpoints, logs, or private paths are included.
- `llm_rerank.py` can read an OpenAI key from `--api_key` or `OPENAI_API_KEY` if users choose the OpenAI backend; no key value is stored in this package.
- If a local Python environment accidentally loads conflicting user-level packages, prefix commands with `PYTHONNOUSERSITE=1`.
