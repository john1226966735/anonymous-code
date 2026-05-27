"""RePlan v0 model: Dynamic plan-guided GNN exploration (shared dim_reduct).

WARNING: This is v0 (shared dim_reduct). For the final method (v1, residual projection),
use replan_model_v1.py instead. base_model.py should import from replan_model_v1.

Key difference from PlanR:
- Plan steps are generated dynamically by DynamicPlanGenerator during GNN forward pass
- Each layer's plan step is conditioned on previous plan steps + selected relations
- guide = dim_reduct(plan_generator_output)  ← shared MLP, causes signal dilution
- Enables adaptive plan adjustment based on actual exploration path
"""

import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter

from models import GNNLayer
from plan_generator import DynamicPlanGenerator


class RePlanExplore(nn.Module):
    """RePlan: Dynamic plan-guided GNN with autoregressive plan generation."""

    def __init__(self, params, loader):
        super(RePlanExplore, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]
        self.K = params.K
        self.sample_flag = params.sample

        self.emb_dim = getattr(params, 'emb_dim', 3584)
        self.emb_dir = getattr(params, 'emb_dir', '../embedding')

        # Question embeddings + dim reduction
        self.question_emb = self.load_qemb().detach()
        mid_dim = min(2096, self.emb_dim)
        self.dim_reduct = nn.Sequential(
            nn.Linear(self.emb_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, self.hidden_dim)
        )

        # Relation embeddings
        self.use_lama_rel = 1
        if self.use_lama_rel == 1:
            self.rela_embed = self.load_rel_emb().detach()
        else:
            self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        # ---- RePlan: Dynamic plan generator ----
        self.plan_generator = DynamicPlanGenerator(
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            n_heads=4,
            n_transformer_layers=2,
            dropout=0.1
        )

        # Load pretrained plan generator if available
        pretrained_path = f'results/{loader.task_dir.split("/")[-1]}_plan_generator_best.pt'
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.plan_generator.load_state_dict(state_dict)
            print(f'Loaded pretrained plan generator from {pretrained_path}')
        except Exception as exc:
            print(f'Failed to load pretrained plan generator from {pretrained_path}: {exc}')
            print('Training from scratch')

        # GNN layers
        self.gnn_layers = []
        for i in range(3):
            self.gnn_layers.append(GNNLayer(
                self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel,
                self.use_lama_rel, self.K, self.sample_flag, act=act
            ))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(params.dropout)
        self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.Wq_final = nn.Linear(self.hidden_dim * 2, 1, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, 1)
        )
        self.Wr = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.loop = nn.Parameter(torch.randn(1, self.hidden_dim))

    def load_qemb(self):
        """Load question embeddings."""
        datapath = self.loader.task_dir
        emb_dir = self.emb_dir
        if 'MetaQA/1-hop' in datapath:
            q_train = np.load(f'{emb_dir}/Meta-1m-train.npy')
            q_valid = np.load(f'{emb_dir}/Meta-1m-valid.npy')
            q_test = np.load(f'{emb_dir}/Meta-1m-test.npy')
        elif 'MetaQA/2-hop' in datapath:
            q_train = np.load(f'{emb_dir}/Meta-2m-train.npy')
            q_valid = np.load(f'{emb_dir}/Meta-2m-valid.npy')
            q_test = np.load(f'{emb_dir}/Meta-2m-test.npy')
        elif 'MetaQA/3-hop' in datapath:
            q_train = np.load(f'{emb_dir}/Meta-3m-train.npy')
            q_valid = np.load(f'{emb_dir}/Meta-3m-valid.npy')
            q_test = np.load(f'{emb_dir}/Meta-3m-test.npy')
        elif 'webqsp' in datapath:
            q_train = np.load(f'{emb_dir}/webqsp-train.npy')
            q_valid = np.load(f'{emb_dir}/webqsp-valid.npy')
            q_test = np.load(f'{emb_dir}/webqsp-test.npy')
        elif 'CWQ' in datapath:
            q_train = np.load(f'{emb_dir}/CWQ-train.npy')
            q_valid = np.load(f'{emb_dir}/CWQ-valid.npy')
            q_test = np.load(f'{emb_dir}/CWQ-test.npy')

        q_emb = np.concatenate((q_train, q_valid, q_test))
        return torch.tensor(q_emb, dtype=torch.float32)

    def load_rel_emb(self):
        """Load relation embeddings."""
        datapath = self.loader.task_dir
        emb_dir = self.emb_dir
        if 'MetaQA' in datapath:
            rel_emb = np.load(f'{emb_dir}/Meta-rel.npy')
        elif 'webqsp' in datapath:
            rel_emb = np.load(f'{emb_dir}/webqsp-rel.npy')
        elif 'CWQ' in datapath:
            rel_emb = np.load(f'{emb_dir}/CWQ-rel.npy')
        print('rel_emb shape: ', rel_emb.shape)
        return torch.tensor(rel_emb, dtype=torch.float32)

    def _aggregate_relation_summary(self, edges, alpha, rel_emb, batch_size):
        """Aggregate selected relations into a summary vector for each question.

        Uses attention-weighted mean of relation embeddings per question.

        Args:
            edges: (n_edges, 6) edge tensor after GNN processing
                   [q_idx, old_node_id, rel_id, new_node_id, old_node_idx, new_node_idx]
            alpha: (n_edges, 1) attention weights from GNN layer
            rel_emb: (n_rel*2+1, hidden_dim) processed relation embeddings
            batch_size: number of questions in batch
        Returns:
            rel_summary: (batch_size, hidden_dim)
        """
        rel_ids = edges[:, 2]                    # (n_edges,)
        hr = rel_emb[rel_ids, :]                  # (n_edges, hidden_dim)
        weighted_rel = hr * alpha                  # (n_edges, hidden_dim)
        q_idx = edges[:, 0]                        # (n_edges,)
        rel_summary = scatter(weighted_rel, q_idx, dim=0,
                              dim_size=batch_size, reduce='mean')
        return rel_summary

    def forward(self, subs, qids, mode='train'):
        """Forward pass with dynamic plan generation.

        During training: generates plan steps dynamically, conditions on relation history
        During inference: same as training
        """
        n_qs = len(qids)
        q_sub = subs
        q_id = torch.LongTensor(qids)

        # Question embedding (for node init + final scoring)
        ques_emb = self.question_emb[q_id, :]
        ques_emb = ques_emb.cuda()
        q_id = q_id.cuda()
        q_emb = self.dim_reduct(ques_emb)
        # keep raw emb on CPU, move to GPU only when plan_generator needs it
        q_emb_raw = ques_emb.detach()  # (batch, emb_dim) on GPU

        # Relation embedding (same as PlanR)
        if self.use_lama_rel == 1:
            self.rela_embed = self.rela_embed.cuda()
            rel_emb = self.dim_reduct(self.rela_embed)
            self.rela_embed.cpu()
            rel_emb = rel_emb[0:self.n_rel, :]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop], dim=0)
        else:
            rel_emb = self.rela_embed

        # Node initialization (same as PlanR: use question embedding)
        n_node = sum(len(sublist) for sublist in subs)
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),
            np.concatenate(subs)
        ]).reshape(2, -1)
        nodes = np.array(nodes, dtype=np.int64)
        nodes = torch.LongTensor(nodes).T.cuda()

        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        hidden = q_emb[nodes[:, 0], :]  # init with question embedding

        # Dynamic plan generation: track plan steps and relation history
        plan_steps = []  # list of (batch, emb_dim) tensors
        rel_history = []  # list of (batch, hidden_dim) tensors

        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))

        # GNN propagation with dynamic plan generation
        for i in range(self.n_layer):
            # Generate next plan step
            if len(plan_steps) == 0:
                next_plan_raw = self.plan_generator(q_emb_raw, plan_history=None, rel_history=None)
            else:
                plan_hist = torch.stack(plan_steps, dim=1)
                rel_hist = torch.stack(rel_history, dim=1) if rel_history else None
                next_plan_raw = self.plan_generator(q_emb_raw, plan_history=plan_hist, rel_history=rel_hist)

            plan_steps.append(next_plan_raw)

            # Project plan step to hidden_dim
            guide_emb = self.dim_reduct(next_plan_raw)  # (batch, hidden_dim)

            # Get neighbors for this layer
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids)

            # GNN layer forward
            num_node, num_edge, hidden, alpha, nodes, edges, old_nodes_new_idx = \
                self.gnn_layers[i](q_sub, q_id, guide_emb, rel_emb, hidden, edges, nodes, old_nodes_new_idx)

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            num_nodes[i, :] += num_node
            num_edges[i, :] += num_edge

            # Aggregate relation summary for next plan step generation
            rel_summary = self._aggregate_relation_summary(edges, alpha, rel_emb, n_qs)
            rel_history.append(rel_summary)

        # Free raw embeddings early
        del q_emb_raw, plan_steps

        # Final scoring (same as PlanR: use question embedding)
        h_qs = q_emb[nodes[:, 0], :]
        scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores

        if mode == 'train':
            return num_nodes, num_edges, scores_all
        else:
            return scores_all

    def change_loader(self, loader):
        self.loader = loader
        self.n_rel = loader.n_rel

    def visual_path(self, subs, rels, objs, filepath='path.txt', mode='test'):
        """Generate reasoning paths for visualization (same as PlanR)."""
        # TODO: implement path visualization with dynamic plans
        pass
