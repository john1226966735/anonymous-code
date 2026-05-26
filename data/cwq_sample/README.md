# CWQ Sample Data

This directory contains a tiny CWQ-style sample dataset for code review.

Schema for each split file:

```json
{
  "id": "cwq_train_0",
  "question": "...",
  "topic_entity": "...",
  "answers": ["..."],
  "graph": [["head", "relation", "tail"], ...],
  "plan_steps": ["...", "..."]
}
```

The sample data is intentionally small and synthetic, but it matches the paper's method flow:
planner pretraining, joint retriever training, test-time path export, and evaluation.

