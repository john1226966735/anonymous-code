import argparse
import json
import os
import site
import sys
from pathlib import Path

# Keep the reviewer environment isolated from a conflicting user-site PyTorch
# installation when this package is run with plain `python -m replan.cli`.
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import torch

from .data import CWQSampleDataset
from .eval import evaluate_predictions, load_jsonl
from .model import RePlanModel


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_pretrain(args):
    dataset = CWQSampleDataset(args.data)
    model = RePlanModel(dim=args.dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    model.to(args.device)
    opt = torch.optim.Adam(model.plan_generator.parameters(), lr=args.lr)
    train_set = dataset.split("train")
    valid_set = dataset.split("valid")
    best = None
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for sample in train_set:
            loss = model.planner_pretrain_loss(sample)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for sample in valid_set:
                vloss += float(model.planner_pretrain_loss(sample).cpu())
            vloss /= max(1, len(valid_set))
        print(f"[pretrain] epoch={epoch} train_loss={total/max(1,len(train_set)):.4f} val_loss={vloss:.4f}")
        if vloss < best_val:
            best_val = vloss
            best = {k: v.cpu() for k, v in model.plan_generator.state_dict().items()}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best, args.output)
            print(f"[pretrain] saved best planner to {args.output}")


def run_train(args):
    dataset = CWQSampleDataset(args.data)
    model = RePlanModel(dim=args.dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    model.to(args.device)
    if args.planner_ckpt:
        state = torch.load(args.planner_ckpt, map_location=args.device)
        model.plan_generator.load_state_dict(state)
        print(f"[train] loaded planner checkpoint from {args.planner_ckpt}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_set = dataset.split("train")
    valid_set = dataset.split("valid")
    best_state = None
    best_val = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for sample in train_set:
            loss, _ = model.retrieval_loss(sample)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            hits = 0
            for sample in valid_set:
                _, cache = model.retrieval_loss(sample)
                top1 = cache.entity_names[int(torch.argmax(cache.candidate_scores).item())]
                hits += int(top1 in sample.answers)
            val_h1 = hits / max(1, len(valid_set))
        print(f"[train] epoch={epoch} train_loss={total/max(1,len(train_set)):.4f} val_h1={val_h1:.4f}")
        if val_h1 >= best_val:
            best_val = val_h1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, args.output)
            print(f"[train] saved best model to {args.output}")


def run_test(args):
    dataset = CWQSampleDataset(args.data)
    model = RePlanModel(dim=args.dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers)
    model.to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state)
    model.eval()
    rows = []
    for sample in dataset.split(args.split):
        with torch.no_grad():
            _, cache = model.retrieval_loss(sample)
            top_candidates = model.top_candidates(cache, k=3)
            pred_answer = top_candidates[0] if top_candidates else ""
            path = model.recover_path(sample, cache, pred_answer)
            rows.append({
                "id": sample.id,
                "question": sample.question,
                "topic_entity": sample.topic_entity,
                "gold_answers": sample.answers,
                "top_candidates": top_candidates,
                "top1": top_candidates[0] if top_candidates else "",
                "pred_answer": pred_answer,
                "recovered_path": path,
            })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output, rows)
    print(f"[test] saved predictions to {args.output}")


def run_evaluate(args):
    rows = load_jsonl(args.predictions)
    metrics = evaluate_predictions(rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def build_parser():
    parser = argparse.ArgumentParser(description="RePlan submission code")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--data", default="data/cwq_sample")
        p.add_argument("--dim", type=int, default=128)
        p.add_argument("--hidden-dim", type=int, default=64)
        p.add_argument("--n-layers", type=int, default=3)
        p.add_argument("--device", default="cpu")

    p = sub.add_parser("pretrain")
    add_common(p)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", default="checkpoints/planner.pt")

    p = sub.add_parser("train")
    add_common(p)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--planner-ckpt", default="checkpoints/planner.pt")
    p.add_argument("--output", default="checkpoints/replan.pt")

    p = sub.add_parser("test")
    add_common(p)
    p.add_argument("--checkpoint", default="checkpoints/replan.pt")
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")
    p.add_argument("--output", default="outputs/test_predictions.jsonl")

    p = sub.add_parser("evaluate")
    p.add_argument("--predictions", default="outputs/test_predictions.jsonl")
    return parser


def main():
    args = build_parser().parse_args()
    if args.cmd == "pretrain":
        run_pretrain(args)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "test":
        run_test(args)
    elif args.cmd == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
