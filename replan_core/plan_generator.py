"""DynamicPlanGenerator: autoregressive plan step generator for RePlan.

Pre-training: given [question_emb, plan_step_0, ..., plan_step_{i-1}], predict plan_step_i
Joint training: given [question_emb, plan_step_0, rel_0, ..., plan_step_{i-1}, rel_{i-1}], predict plan_step_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPlanGenerator(nn.Module):
    """Lightweight autoregressive model that generates plan step embeddings.

    During pre-training, it learns to predict the next plan step from previous steps.
    During joint training with GNN, it also conditions on selected relations.
    """

    def __init__(self, emb_dim, hidden_dim, n_heads=4, n_transformer_layers=2, dropout=0.1):
        """
        Args:
            emb_dim: dimension of plan step embeddings (e.g. 3584 for Qwen2.5-7B)
            hidden_dim: internal dimension (e.g. 256, same as GNN hidden_dim)
            n_heads: number of attention heads in Transformer
            n_transformer_layers: number of Transformer encoder layers
            dropout: dropout rate
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Input projections (from different sources to unified hidden_dim)
        self.proj_question = nn.Linear(emb_dim, hidden_dim)
        self.proj_plan = nn.Linear(emb_dim, hidden_dim)
        self.proj_relation = nn.Linear(hidden_dim, hidden_dim)  # relation is already hidden_dim

        # Type embeddings: 0=question, 1=plan_step, 2=relation
        self.type_embed = nn.Embedding(3, hidden_dim)

        # Positional encoding (simple learnable, max 16 positions)
        self.pos_embed = nn.Embedding(16, hidden_dim)

        # Transformer encoder (causal masking applied in forward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # Output projection: hidden_dim -> emb_dim (predict plan step in original space)
        self.output_proj = nn.Linear(hidden_dim, emb_dim)

        # Layer norm
        self.ln = nn.LayerNorm(hidden_dim)

    def _build_sequence(self, question_emb, plan_history=None, rel_history=None):
        """Build input sequence for the Transformer.

        Args:
            question_emb: (batch, emb_dim)
            plan_history: (batch, n_steps, emb_dim) or None
            rel_history: (batch, n_rels, hidden_dim) or None
                         n_rels <= n_steps (relation for step i is available after GNN layer i)
        Returns:
            seq: (batch, seq_len, hidden_dim)
        """
        batch_size = question_emb.size(0)
        device = question_emb.device
        tokens = []
        type_ids = []

        # Token 0: question
        q = self.proj_question(question_emb)  # (batch, hidden_dim)
        tokens.append(q)
        type_ids.append(0)

        # Interleave plan steps and relations
        if plan_history is not None:
            n_steps = plan_history.size(1)
            n_rels = rel_history.size(1) if rel_history is not None else 0

            for i in range(n_steps):
                p = self.proj_plan(plan_history[:, i, :])  # (batch, hidden_dim)
                tokens.append(p)
                type_ids.append(1)

                if i < n_rels:
                    r = self.proj_relation(rel_history[:, i, :])  # (batch, hidden_dim)
                    tokens.append(r)
                    type_ids.append(2)

        # Stack tokens
        seq = torch.stack(tokens, dim=1)  # (batch, seq_len, hidden_dim)

        # Add type embeddings
        type_tensor = torch.tensor(type_ids, device=device).unsqueeze(0).expand(batch_size, -1)
        seq = seq + self.type_embed(type_tensor)

        # Add positional embeddings
        pos_ids = torch.arange(seq.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
        seq = seq + self.pos_embed(pos_ids)

        return self.ln(seq)

    def forward(self, question_emb, plan_history=None, rel_history=None):
        """Generate the next plan step embedding.

        Args:
            question_emb: (batch, emb_dim)
            plan_history: (batch, n_steps, emb_dim) or None (for generating step 0)
            rel_history: (batch, n_rels, hidden_dim) or None (pre-training mode)
        Returns:
            next_plan_step: (batch, emb_dim)
        """
        seq = self._build_sequence(question_emb, plan_history, rel_history)

        # Causal mask: each position can only attend to itself and previous positions
        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device), diagonal=1
        ).bool()

        # Transformer forward
        output = self.transformer(seq, mask=causal_mask)  # (batch, seq_len, hidden_dim)

        # Take the last position's output
        last_hidden = output[:, -1, :]  # (batch, hidden_dim)

        # Project to plan step embedding space
        next_plan_step = self.output_proj(last_hidden)  # (batch, emb_dim)

        return next_plan_step

    def generate_all_steps(self, question_emb, max_steps, rel_callback=None):
        """Autoregressively generate all plan steps.

        Used during GNN inference: generate step i, run GNN layer i, get relation,
        then generate step i+1 conditioned on the relation.

        Args:
            question_emb: (batch, emb_dim)
            max_steps: maximum number of steps to generate
            rel_callback: optional function(step_idx, plan_step) -> rel_emb
                          Called after each step to get the relation embedding
                          from GNN. If None, generates without relation conditioning.
        Returns:
            plan_steps: list of (batch, emb_dim) tensors
        """
        plan_steps = []
        rel_history = []

        for i in range(max_steps):
            # Build plan history
            if len(plan_steps) == 0:
                plan_hist = None
                rel_hist = None
            else:
                plan_hist = torch.stack(plan_steps, dim=1)  # (batch, i, emb_dim)
                rel_hist = torch.stack(rel_history, dim=1) if rel_history else None

            # Generate next plan step
            next_step = self.forward(question_emb, plan_hist, rel_hist)
            plan_steps.append(next_step)

            # Get relation from GNN (if callback provided)
            if rel_callback is not None:
                rel_emb = rel_callback(i, next_step)  # (batch, hidden_dim)
                rel_history.append(rel_emb)

        return plan_steps
