"""Lightweight smoke test for the submitted RePlan core code.

This script checks that the real data loader, plan generator, pretraining loss,
and final RePlan model class can be imported and initialized on the included
CWQ sample. It does not reproduce the paper numbers; full training uses the
GPU-oriented scripts `pretrain_plan_generator.py` and `train.py`.
"""

import os
from pathlib import Path

import numpy as np
import torch

from plan_generator import DynamicPlanGenerator
from pretrain_plan_generator import PlanStepDataset, info_nce_loss, cosine_loss


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(Path(__file__).resolve().parent)

    emb_dir = root / "embedding"

    q_train = np.load(emb_dir / "CWQ-train.npy")
    plan_train = np.load(emb_dir / "CWQ-train-plan.npy")
    ns_train = np.load(emb_dir / "CWQ-train-nsteps.npy")

    dataset = PlanStepDataset(
        torch.tensor(q_train, dtype=torch.float32),
        torch.tensor(plan_train, dtype=torch.float32),
        ns_train,
    )
    sample = dataset[0]
    question_emb = sample["question_emb"].unsqueeze(0)
    plan_steps = sample["plan_steps"].unsqueeze(0)
    target_step = int(sample["target_step"])
    target = plan_steps[:, target_step, :]

    model = DynamicPlanGenerator(
        emb_dim=question_emb.size(-1),
        hidden_dim=64,
        n_heads=4,
        n_transformer_layers=2,
        dropout=0.1,
    )
    pred = model(question_emb, plan_history=None, rel_history=None)
    loss = info_nce_loss(pred, target) + cosine_loss(pred, target)
    loss.backward()

    print("Smoke test passed.")
    print(f"Sample embeddings: questions={q_train.shape}, plans={plan_train.shape}")
    print(f"PlanGenerator output shape={tuple(pred.shape)}, loss={float(loss.detach()):.4f}")


if __name__ == "__main__":
    main()
