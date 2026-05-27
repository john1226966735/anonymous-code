"""Pre-train the DynamicPlanGenerator on existing plan step embeddings.

Training data: LLM-generated plan step embeddings (already available).
Task: given [question_emb, plan_step_0, ..., plan_step_{i-1}], predict plan_step_i.
Loss: InfoNCE (contrastive) + cosine similarity.

Usage:
    python pretrain_plan_generator.py --dataset CWQ --gpu 0 --epochs 50 --batch_size 128
    python pretrain_plan_generator.py --dataset CWQ --gpu 0 --epochs 20 --batch_size 128 --fast
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from plan_generator import DynamicPlanGenerator


class PlanStepDataset(Dataset):
    """Dataset for plan step prediction.

    Each sample is (question_emb, plan_steps, n_steps, target_step_idx).
    For a question with n_steps plan steps, we create n_steps training samples:
      - target_step_idx=0: input=[question], target=plan_step_0
      - target_step_idx=1: input=[question, plan_step_0], target=plan_step_1
      - ...
    """

    def __init__(self, question_emb, plan_emb, nsteps):
        """
        Args:
            question_emb: (N, emb_dim) question embeddings
            plan_emb: (N, max_steps, emb_dim) plan step embeddings
            nsteps: (N,) number of valid plan steps per question
        """
        self.question_emb = question_emb
        self.plan_emb = plan_emb
        self.nsteps = nsteps

        # Build index: (question_idx, target_step_idx)
        self.samples = []
        for i in range(len(nsteps)):
            for step in range(nsteps[i]):
                self.samples.append((i, step))

        print(f'PlanStepDataset: {len(question_emb)} questions, '
              f'{len(self.samples)} training samples, '
              f'nsteps distribution: {dict(zip(*np.unique(nsteps, return_counts=True)))}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q_idx, target_step = self.samples[idx]
        return {
            'question_emb': self.question_emb[q_idx],       # (emb_dim,)
            'plan_steps': self.plan_emb[q_idx],              # (max_steps, emb_dim)
            'n_steps': self.nsteps[q_idx],                   # scalar
            'target_step': target_step,                       # scalar
            'q_idx': q_idx,                                   # for debugging
        }


def info_nce_loss(predicted, target, temperature=0.07):
    """InfoNCE contrastive loss.

    Positive: target plan step.
    Negatives: all other plan steps in the batch.

    Args:
        predicted: (batch, emb_dim)
        target: (batch, emb_dim)
        temperature: temperature for softmax
    Returns:
        loss: scalar
    """
    # Normalize
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)

    # Similarity matrix: (batch, batch)
    logits = torch.mm(predicted, target.t()) / temperature

    # Labels: diagonal (each sample's positive is itself)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def cosine_loss(predicted, target):
    """1 - cosine_similarity, averaged over batch."""
    return 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_cos_sim = 0
    n_batches = 0

    for batch in dataloader:
        question_emb = batch['question_emb'].to(device)      # (B, emb_dim)
        plan_steps = batch['plan_steps'].to(device)           # (B, max_steps, emb_dim)
        target_step_idx = batch['target_step'].to(device)     # (B,)

        # Build plan history: steps before target
        # For each sample, plan_history = plan_steps[:target_step_idx]
        # Since target_step_idx varies per sample, we need to handle this carefully
        batch_size = question_emb.size(0)
        max_target = target_step_idx.max().item()

        # Gather target embeddings
        target_emb = plan_steps[torch.arange(batch_size), target_step_idx]  # (B, emb_dim)

        # Build plan history for each sample
        # We pad to max_target steps, masking out invalid positions
        if max_target == 0:
            # Predicting step 0: no plan history
            predicted = model(question_emb, plan_history=None, rel_history=None)
        else:
            # Build plan history tensor: (B, max_target, emb_dim)
            plan_history = plan_steps[:, :max_target, :]  # (B, max_target, emb_dim)

            # Zero out steps beyond each sample's target
            for i in range(batch_size):
                t = target_step_idx[i].item()
                if t < max_target:
                    plan_history[i, t:, :] = 0

            predicted = model(question_emb, plan_history=plan_history, rel_history=None)

        # Combined loss: InfoNCE + cosine
        loss_nce = info_nce_loss(predicted, target_emb)
        loss_cos = cosine_loss(predicted, target_emb)
        loss = loss_nce + loss_cos

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            cos_sim = F.cosine_similarity(predicted, target_emb, dim=-1).mean().item()
        total_cos_sim += cos_sim
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_cos = total_cos_sim / n_batches
    return avg_loss, avg_cos


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_cos_sim = 0
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        question_emb = batch['question_emb'].to(device)
        plan_steps = batch['plan_steps'].to(device)
        target_step_idx = batch['target_step'].to(device)

        batch_size = question_emb.size(0)
        max_target = target_step_idx.max().item()
        target_emb = plan_steps[torch.arange(batch_size), target_step_idx]

        if max_target == 0:
            predicted = model(question_emb, plan_history=None, rel_history=None)
        else:
            plan_history = plan_steps[:, :max_target, :].clone()
            for i in range(batch_size):
                t = target_step_idx[i].item()
                if t < max_target:
                    plan_history[i, t:, :] = 0
            predicted = model(question_emb, plan_history=plan_history, rel_history=None)

        loss_nce = info_nce_loss(predicted, target_emb)
        loss_cos = cosine_loss(predicted, target_emb)
        loss = loss_nce + loss_cos

        cos_sim = F.cosine_similarity(predicted, target_emb, dim=-1).mean().item()
        total_cos_sim += cos_sim
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches, total_cos_sim / n_batches


def main():
    parser = argparse.ArgumentParser(description="Pre-train DynamicPlanGenerator")
    parser.add_argument('--dataset', type=str, default='CWQ', choices=['CWQ', 'webqsp'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--emb_dim', type=int, default=3584)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fast', action='store_true', help='Fast mode: fewer epochs, subsample data')
    parser.add_argument('--emb_dir', type=str, default='../embedding')
    parser.add_argument('--question_emb_dir', type=str, default=None,
                        help='Directory for question embeddings; defaults to --emb_dir')
    parser.add_argument('--plan_emb_dir', type=str, default=None,
                        help='Directory for plan-step embeddings; defaults to --emb_dir')
    parser.add_argument('--output_ckpt', type=str, default=None,
                        help='Output checkpoint path; defaults to results/{dataset}_plan_generator_best.pt')
    parser.add_argument('--perf_file', type=str, default=None,
                        help='Output metrics log path; defaults to results/{dataset}_plan_generator_perf.txt')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Load data
    emb_dir = args.emb_dir
    question_emb_dir = args.question_emb_dir or emb_dir
    plan_emb_dir = args.plan_emb_dir or emb_dir
    ds = args.dataset

    print(f'Loading {ds} question embeddings from {question_emb_dir}...')
    print(f'Loading {ds} plan embeddings from {plan_emb_dir}...')

    # Load question embeddings (same as PlanR/DualR)
    if ds == 'CWQ':
        q_train = np.load(f'{question_emb_dir}/CWQ-train.npy')
        q_valid = np.load(f'{question_emb_dir}/CWQ-valid.npy')
    elif ds == 'webqsp':
        q_train = np.load(f'{question_emb_dir}/webqsp-train.npy')
        q_valid = np.load(f'{question_emb_dir}/webqsp-valid.npy')

    # Load plan embeddings
    plan_train = np.load(f'{plan_emb_dir}/{ds}-train-plan.npy')
    plan_valid = np.load(f'{plan_emb_dir}/{ds}-valid-plan.npy')
    ns_train = np.load(f'{plan_emb_dir}/{ds}-train-nsteps.npy')
    ns_valid = np.load(f'{plan_emb_dir}/{ds}-valid-nsteps.npy')

    print(f'Train: q={q_train.shape}, plan={plan_train.shape}, nsteps={ns_train.shape}')
    print(f'Valid: q={q_valid.shape}, plan={plan_valid.shape}, nsteps={ns_valid.shape}')

    # Fast mode
    n_epochs = args.epochs
    if args.fast:
        n_epochs = min(20, args.epochs)
        # Subsample train to 1/4
        n = len(q_train) // 4
        idx = np.random.permutation(len(q_train))[:n]
        q_train = q_train[idx]
        plan_train = plan_train[idx]
        ns_train = ns_train[idx]
        print(f'[FAST] train subsampled to {n} questions')

    # Create datasets
    train_dataset = PlanStepDataset(
        torch.tensor(q_train, dtype=torch.float32),
        torch.tensor(plan_train, dtype=torch.float32),
        ns_train
    )
    valid_dataset = PlanStepDataset(
        torch.tensor(q_valid, dtype=torch.float32),
        torch.tensor(plan_valid, dtype=torch.float32),
        ns_valid
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Create model
    model = DynamicPlanGenerator(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_transformer_layers=args.n_layers,
        dropout=0.1
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,} ({n_params/1e6:.1f}M)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    os.makedirs('results', exist_ok=True)
    perf_file = args.perf_file or f'results/{ds}_plan_generator_perf.txt'
    output_ckpt = args.output_ckpt or f'results/{ds}_plan_generator_best.pt'
    best_cos = -1
    best_epoch = -1

    print(f'\nStarting pre-training for {n_epochs} epochs...')
    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, train_cos = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_cos = evaluate(model, valid_loader, device)
        scheduler.step()
        t1 = time.time()

        line = (f'epoch {epoch:3d} | train_loss={train_loss:.4f} train_cos={train_cos:.4f} | '
                f'val_loss={val_loss:.4f} val_cos={val_cos:.4f} | time={t1-t0:.1f}s')
        print(line)

        with open(perf_file, 'a') as f:
            f.write(line + '\n')

        if val_cos > best_cos:
            best_cos = val_cos
            best_epoch = epoch
            torch.save(model.state_dict(), output_ckpt)

    print(f'\nBest val cosine similarity: {best_cos:.4f} at epoch {best_epoch}')
    print(f'Model saved to {output_ckpt}')


if __name__ == '__main__':
    main()
