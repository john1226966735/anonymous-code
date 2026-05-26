from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Sample
from .text import hash_embed, batch_hash_embed


@dataclass
class ForwardCache:
    entity_names: List[str]
    relation_names: List[str]
    edges_by_layer: List[List[Tuple[int, int, int, float]]]
    candidate_scores: torch.Tensor
    candidate_indices: torch.Tensor


class DynamicPlanGenerator(nn.Module):
    def __init__(self, emb_dim: int = 128, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.proj_question = nn.Linear(emb_dim, hidden_dim)
        self.proj_plan = nn.Linear(emb_dim, hidden_dim)
        self.proj_relation = nn.Linear(hidden_dim, hidden_dim)
        self.type_embed = nn.Embedding(3, hidden_dim)
        self.pos_embed = nn.Embedding(16, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def _build_seq(
        self,
        question_emb: torch.Tensor,
        plan_history: Optional[torch.Tensor],
        rel_history: Optional[torch.Tensor],
    ) -> torch.Tensor:
        tokens = [self.proj_question(question_emb)]
        type_ids = [0]
        if plan_history is not None:
            n_steps = plan_history.size(1)
            n_rels = 0 if rel_history is None else rel_history.size(1)
            for i in range(n_steps):
                tokens.append(self.proj_plan(plan_history[:, i, :]))
                type_ids.append(1)
                if i < n_rels:
                    tokens.append(self.proj_relation(rel_history[:, i, :]))
                    type_ids.append(2)
        seq = torch.stack(tokens, dim=1)
        type_tensor = torch.tensor(type_ids, device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
        pos_tensor = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
        seq = seq + self.type_embed(type_tensor) + self.pos_embed(pos_tensor)
        return self.ln(seq)

    def forward(
        self,
        question_emb: torch.Tensor,
        plan_history: Optional[torch.Tensor] = None,
        rel_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq = self._build_seq(question_emb, plan_history, rel_history)
        seq_len = seq.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()
        out = self.transformer(seq, mask=mask)
        return self.out_proj(out[:, -1, :])


class RePlanModel(nn.Module):
    def __init__(self, dim: int = 128, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.plan_generator = DynamicPlanGenerator(emb_dim=dim, hidden_dim=hidden_dim)
        self.question_proj = nn.Linear(dim, hidden_dim)
        self.plan_delta_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relation_proj = nn.Linear(dim, hidden_dim)
        self.entity_proj = nn.Linear(dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_entities(self, entity_names: List[str]) -> torch.Tensor:
        return self.entity_proj(batch_hash_embed(entity_names, dim=self.dim).to(next(self.parameters()).device))

    def _encode_relations(self, relation_names: List[str]) -> torch.Tensor:
        return self.relation_proj(batch_hash_embed(relation_names, dim=self.dim).to(next(self.parameters()).device))

    def _encode_question(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = hash_embed(question, dim=self.dim).to(next(self.parameters()).device)
        return raw, self.question_proj(raw)

    def _encode_plan_steps(self, steps: List[str]) -> torch.Tensor:
        if not steps:
            return torch.zeros(0, self.dim, device=next(self.parameters()).device)
        return batch_hash_embed(steps, dim=self.dim).to(next(self.parameters()).device)

    def _build_graph(self, sample: Sample) -> Tuple[List[str], Dict[str, int], List[Tuple[int, int, int]]]:
        entity_names = sorted({sample.topic_entity, *sample.answers, *[h for h, _, _ in sample.graph], *[t for _, _, t in sample.graph]})
        entity2idx = {name: i for i, name in enumerate(entity_names)}
        relation_names = sorted({r for _, r, _ in sample.graph})
        relation2idx = {name: i for i, name in enumerate(relation_names)}
        edges = [(entity2idx[h], relation2idx[r], entity2idx[t]) for h, r, t in sample.graph]
        return entity_names, relation2idx, edges

    def _plan_guidance(
        self,
        q_raw: torch.Tensor,
        q_hid: torch.Tensor,
        plan_history: Optional[List[torch.Tensor]],
        rel_history: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        if plan_history is None or len(plan_history) == 0:
            next_plan = self.plan_generator(q_raw.unsqueeze(0), None, None).squeeze(0)
        else:
            plan_hist = torch.stack(plan_history, dim=0).unsqueeze(0)
            rel_hist = None if rel_history is None or len(rel_history) == 0 else torch.stack(rel_history, dim=0).unsqueeze(0)
            next_plan = self.plan_generator(q_raw.unsqueeze(0), plan_hist, rel_hist).squeeze(0)
        guide = q_hid + self.plan_delta_proj(next_plan - q_raw)
        return next_plan, guide

    def forward(self, sample: Sample, return_cache: bool = False):
        device = next(self.parameters()).device
        q_raw, q_hid = self._encode_question(sample.question)
        entity_names, relation2idx, edges = self._build_graph(sample)
        entity_hid = self._encode_entities(entity_names)
        relation_names = [name for name, _ in sorted(relation2idx.items(), key=lambda x: x[1])]
        relation_hid = self._encode_relations(relation_names)

        topic_idx = entity_names.index(sample.topic_entity)
        entity_hid = entity_hid.clone()
        entity_hid[topic_idx] = entity_hid[topic_idx] + q_hid

        plan_history: List[torch.Tensor] = []
        rel_history: List[torch.Tensor] = []
        edges_by_layer = []

        for layer in range(self.n_layers):
            next_plan, guide = self._plan_guidance(q_raw, q_hid, plan_history, rel_history)
            plan_history.append(next_plan)

            messages = torch.zeros_like(entity_hid)
            attn_records: List[Tuple[int, int, int, float]] = []

            for src_idx, rel_idx, dst_idx in edges:
                h_s = entity_hid[src_idx]
                h_r = relation_hid[rel_idx]
                logit = self.edge_mlp(torch.cat([h_s, h_r, guide, h_r * guide], dim=-1))
                attn = torch.sigmoid(logit).squeeze()
                msg = attn * (h_s + h_r)
                messages[dst_idx] = messages[dst_idx] + msg
                attn_records.append((src_idx, rel_idx, dst_idx, float(attn.detach().cpu())))

            entity_hid = self.gru(messages, entity_hid)
            edges_by_layer.append(attn_records)

            if attn_records:
                rel_ids = torch.tensor([x[1] for x in attn_records], device=device)
                attn_vals = torch.tensor([x[3] for x in attn_records], device=device).unsqueeze(-1)
                rel_summary = (attn_vals * relation_hid[rel_ids]).sum(dim=0) / (attn_vals.sum() + 1e-8)
            else:
                rel_summary = torch.zeros(self.hidden_dim, device=device)
            rel_history.append(rel_summary)

        scores = self.score_mlp(torch.cat([entity_hid, q_hid.expand(entity_hid.size(0), -1)], dim=-1)).squeeze(-1)
        if return_cache:
            cache = ForwardCache(
                entity_names=entity_names,
                relation_names=relation_names,
                edges_by_layer=edges_by_layer,
                candidate_scores=scores.detach().cpu(),
                candidate_indices=torch.arange(len(entity_names)),
            )
            return scores, cache
        return scores

    def planner_pretrain_loss(self, sample: Sample) -> torch.Tensor:
        q_raw, _ = self._encode_question(sample.question)
        gold_steps = self._encode_plan_steps(sample.plan_steps)
        losses = []
        plan_history: List[torch.Tensor] = []
        rel_history: List[torch.Tensor] = []
        for i in range(gold_steps.size(0)):
            if i == 0:
                pred = self.plan_generator(q_raw.unsqueeze(0), None, None).squeeze(0)
            else:
                plan_hist = torch.stack(plan_history, dim=0).unsqueeze(0)
                rel_hist = None if len(rel_history) == 0 else torch.stack(rel_history, dim=0).unsqueeze(0)
                pred = self.plan_generator(q_raw.unsqueeze(0), plan_hist, rel_hist).squeeze(0)
            target = gold_steps[i]
            plan_history.append(target.detach())
            losses.append(1.0 - F.cosine_similarity(pred, target, dim=0))
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(losses).mean()

    def retrieval_loss(self, sample: Sample) -> Tuple[torch.Tensor, ForwardCache]:
        scores, cache = self.forward(sample, return_cache=True)
        entity_names = cache.entity_names
        gold = set(sample.answers)
        gold_indices = [i for i, name in enumerate(entity_names) if name in gold]
        if not gold_indices:
            gold_indices = [entity_names.index(sample.topic_entity)]
        log_probs = F.log_softmax(scores, dim=0)
        loss = -torch.logsumexp(log_probs[gold_indices], dim=0)
        return loss, cache

    @staticmethod
    def top_candidates(cache: ForwardCache, k: int = 3) -> List[str]:
        topk = torch.topk(cache.candidate_scores, k=min(k, len(cache.entity_names)))
        return [cache.entity_names[i] for i in topk.indices.tolist()]

    def recover_path(self, sample: Sample, cache: ForwardCache, candidate: str) -> List[Tuple[str, str, str]]:
        entity_names = cache.entity_names
        rel_names = cache.relation_names
        entity2idx = {n: i for i, n in enumerate(entity_names)}
        if candidate not in entity2idx:
            return []
        current = entity2idx[candidate]
        path: List[Tuple[str, str, str]] = []
        for layer in reversed(range(len(cache.edges_by_layer))):
            incoming = [edge for edge in cache.edges_by_layer[layer] if edge[2] == current]
            if not incoming:
                continue
            src_idx, rel_idx, dst_idx, _ = max(incoming, key=lambda x: x[3])
            path.append((entity_names[src_idx], rel_names[rel_idx], entity_names[dst_idx]))
            current = src_idx
        path.reverse()
        return path

