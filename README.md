# RePlan Core Code

This directory contains a compact, runnable implementation of the core RePlan pipeline described in the paper. It is intended for anonymous review and includes a small CWQ-style sample dataset.

## What Is Included

- `replan/model.py`: dynamic latent plan generator, plan-to-guidance conversion, GNN graph propagation, entity scoring, and path recovery.
- `replan/cli.py`: command-line entry points for planner pretraining, joint retriever training, testing, and evaluation.
- `replan/data.py`: CWQ-style data loader.
- `replan/eval.py`: Entity Hits@1 and answer Hits@1 evaluation.
- `replan/text.py`: deterministic hashing text encoder used by the sample run.
- `data/cwq_sample/`: small CWQ-style train/valid/test files.

The sample package uses deterministic hashing embeddings instead of downloading an external text embedding model. This keeps the reviewer run lightweight while preserving the method flow: offline plan supervision, dynamic plan-guided graph retrieval, path export, and evaluation.

## Environment

The code requires Python 3.8+ and PyTorch.

```bash
cd EMNLP2026_submission_code
pip install -r requirements.txt
```

If your Python environment has conflicting user-level packages, run the commands below with `PYTHONNOUSERSITE=1`.

## Run The Full Pipeline

Planner pretraining:

```bash
python -m replan.cli pretrain \
  --data data/cwq_sample \
  --output checkpoints/planner.pt \
  --epochs 5
```

Joint retriever training:

```bash
python -m replan.cli train \
  --data data/cwq_sample \
  --planner-ckpt checkpoints/planner.pt \
  --output checkpoints/replan.pt \
  --epochs 5
```

Test-time retrieval and path export:

```bash
python -m replan.cli test \
  --data data/cwq_sample \
  --checkpoint checkpoints/replan.pt \
  --split test \
  --output outputs/test_predictions.jsonl
```

Evaluation:

```bash
python -m replan.cli evaluate \
  --predictions outputs/test_predictions.jsonl
```

Expected output format:

```json
{
  "Entity Hits@1": 1.0,
  "Answer Hits@1": 1.0,
  "N": 1
}
```

Because the included dataset is intentionally tiny, these numbers are only a sanity check that the code path runs correctly. They are not the paper's reported benchmark results.

## Data Format

Each example contains:

```json
{
  "id": "cwq_train_0",
  "question": "Who directed the movie starring the actor born in Paris?",
  "topic_entity": "Paris",
  "answers": ["Bob"],
  "graph": [["Paris", "birthplace", "Alice"], ["Alice", "starring", "Film_A"]],
  "plan_steps": ["find the actor born in Paris", "find the movie starring that actor"]
}
```

- `graph` is a local question subgraph represented as triples.
- `plan_steps` are offline textual subgoals used to pretrain the latent planner.
- During joint training and testing, the model generates latent plan embeddings internally and does not call an LLM during graph traversal.

## Main Commands

```bash
python -m replan.cli pretrain --help
python -m replan.cli train --help
python -m replan.cli test --help
python -m replan.cli evaluate --help
```

