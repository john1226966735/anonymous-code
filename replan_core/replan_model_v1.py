"""RePlan v1 model (FINAL METHOD): Dynamic plan-guided GNN with residual plan projection.

This is the main model file. base_model.py imports from here.

Key differences from v0 (replan_model.py):
- v0 used shared dim_reduct to project plan steps → signal dilution
- v1 uses residual projection: guide = q_emb + plan_delta_proj(plan_raw - q_raw)
- Independent MLP preserves plan-question difference signal

Components:
- C1: Plan guidance (guide replaces q_emb in GNN attention)
- C2: DynamicPlanGenerator (autoregressive Transformer)
- C3: Residual projection (plan_delta_proj)
- C4: Plan generator pretrained on LLM plan embeddings
- C5: Relation history feedback to plan generator
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
        self.no_rel_feedback = getattr(params, 'no_rel_feedback', False)
        self.projection_mode = getattr(params, 'projection_mode', 'residual')

        # Question embeddings + dim reduction
        self.question_emb = self.load_qemb().detach()
        mid_dim = min(2096, self.emb_dim)
        self.dim_reduct = nn.Sequential(
            nn.Linear(self.emb_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, self.hidden_dim)
        )

        # ---- v1: Residual plan projection ----
        # Independent MLP for projecting plan-question residual
        self.plan_delta_proj = nn.Sequential(
            nn.Linear(self.emb_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, self.hidden_dim)
        )
        if self.projection_mode == 'layer_control':
            # Capacity-control baseline: same residual projection path, but the
            # residual source is only a learned layer embedding, not a plan.
            self.layer_control_raw = nn.Parameter(torch.zeros(self.n_layer, self.emb_dim))

        # Relation embeddings
        self.use_lama_rel = 1
        if self.use_lama_rel == 1:
            self.rela_embed = self.load_rel_emb().detach()
        else:
            self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        # ---- RePlan: Dynamic plan generator ----
        self.plan_generator = None
        if self.projection_mode != 'layer_control':
            self.plan_generator = DynamicPlanGenerator(
                emb_dim=self.emb_dim,
                hidden_dim=self.hidden_dim,
                n_heads=4,
                n_transformer_layers=2,
                dropout=0.1
            )

        # Load pretrained plan generator if available
        pretrained_path = getattr(
            params,
            'plan_generator_ckpt',
            None
        ) or f'results/{loader.task_dir.split("/")[-1]}_plan_generator_best.pt'
        no_pretrain = getattr(params, 'no_pretrain', False)
        freeze_planner = getattr(params, 'freeze_planner', False)

        if self.projection_mode == 'layer_control':
            print('[capacity control] Using learned layer-wise residual guidance without plan generator or plan supervision')
        elif no_pretrain:
            print(f'[A1 ablation] Skipping pretrained plan generator, training from scratch')
        else:
            try:
                state_dict = torch.load(pretrained_path, map_location='cpu')
                self.plan_generator.load_state_dict(state_dict)
                print(f'Loaded pretrained plan generator from {pretrained_path}')
            except Exception as exc:
                print(f'Failed to load pretrained plan generator from {pretrained_path}: {exc}')
                print('Training from scratch')

        # A3 ablation: freeze plan generator parameters
        if freeze_planner and self.plan_generator is not None:
            for param in self.plan_generator.parameters():
                param.requires_grad = False
            print(f'[A3 ablation] Froze plan generator parameters')

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

    def _get_layer_control_guide(self, q_emb, layer_idx):
        """Capacity-control guidance without plan text or DynamicPlanGenerator."""
        batch_size = q_emb.size(0)
        delta_raw = self.layer_control_raw[layer_idx].unsqueeze(0).expand(batch_size, -1)
        delta_proj = self.plan_delta_proj(delta_raw)
        return q_emb + delta_proj

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
            if self.projection_mode == 'layer_control':
                guide_emb = self._get_layer_control_guide(q_emb, i)
            else:
                # Generate next plan step
                if len(plan_steps) == 0:
                    next_plan_raw = self.plan_generator(q_emb_raw, plan_history=None, rel_history=None)
                else:
                    plan_hist = torch.stack(plan_steps, dim=1)
                    # A2 ablation: disable relation history feedback
                    rel_hist = None if self.no_rel_feedback else (torch.stack(rel_history, dim=1) if rel_history else None)
                    next_plan_raw = self.plan_generator(q_emb_raw, plan_history=plan_hist, rel_history=rel_hist)

                plan_steps.append(next_plan_raw)

                # v1: Residual plan projection.
                # additive is the residual-null control: keep the independent
                # projection head but remove the plan-question subtraction.
                if self.projection_mode == 'additive':
                    delta = next_plan_raw
                else:
                    delta = next_plan_raw - q_emb_raw      # (batch, emb_dim)
                delta_proj = self.plan_delta_proj(delta)    # (batch, hidden_dim)
                guide_emb = q_emb + delta_proj              # (batch, hidden_dim)

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

            # Aggregate relation summary for next plan step generation.
            if self.projection_mode != 'layer_control':
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

    def visual_path(self, subs, rels, objs, batch_idx, filepath='path.txt', mode='test'):
        """Generate reasoning paths for visualization with dynamic plan generation."""
        n_qs = len(rels)
        q_sub = subs
        q_id = torch.LongTensor(rels)

        # Question embedding
        ques_emb = self.question_emb[q_id, :]
        ques_emb = ques_emb.cuda()
        q_id = q_id.cuda()
        q_emb = self.dim_reduct(ques_emb)
        q_emb_raw = ques_emb.detach()

        # Relation embedding
        if self.use_lama_rel == 1:
            self.rela_embed = self.rela_embed.cuda()
            rel_emb = self.dim_reduct(self.rela_embed)
            self.rela_embed.cpu()
            rel_emb = rel_emb[0:self.n_rel, :]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop], dim=0)
        else:
            rel_emb = self.rela_embed

        # Node initialization
        n_node = sum(len(sublist) for sublist in subs)
        nodes = np.concatenate([
            np.repeat(np.arange(len(subs)), [len(sublist) for sublist in subs]),
            np.concatenate(subs)
        ]).reshape(2, -1)
        nodes = np.array(nodes, dtype=np.int64)
        nodes = torch.LongTensor(nodes).T.cuda()

        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        hidden = q_emb[nodes[:, 0], :]

        # Track all nodes, edges, weights for path reconstruction
        all_nodes = []
        all_edges = []
        all_weights = []

        # Track dynamic plan generation
        plan_steps = []
        rel_history = []

        # GNN propagation with dynamic plan generation
        for i in range(self.n_layer):
            if self.projection_mode == 'layer_control':
                guide_emb = self._get_layer_control_guide(q_emb, i)
            else:
                # Generate next plan step (same as forward)
                if len(plan_steps) == 0:
                    next_plan_raw = self.plan_generator(q_emb_raw, plan_history=None, rel_history=None)
                else:
                    plan_hist = torch.stack(plan_steps, dim=1)
                    rel_hist = None if self.no_rel_feedback else (torch.stack(rel_history, dim=1) if rel_history else None)
                    next_plan_raw = self.plan_generator(q_emb_raw, plan_history=plan_hist, rel_history=rel_hist)

                plan_steps.append(next_plan_raw)

                if self.projection_mode == 'additive':
                    delta = next_plan_raw
                else:
                    delta = next_plan_raw - q_emb_raw
                delta_proj = self.plan_delta_proj(delta)
                guide_emb = q_emb + delta_proj

            # Get neighbors and run GNN layer
            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), rels)
            num_node, num_edge, hidden, weights, nodes, edges, old_nodes_new_idx = \
                self.gnn_layers[i](q_sub, q_id, guide_emb, rel_emb, hidden, edges, nodes, old_nodes_new_idx)

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            # Save for path reconstruction
            all_nodes.append(nodes.cpu().data.numpy())
            all_edges.append(edges.cpu().data.numpy())
            all_weights.append(weights.cpu().data.numpy())

            # Aggregate relation summary for next plan step.
            if self.projection_mode != 'layer_control':
                rel_summary = self._aggregate_relation_summary(edges, weights, rel_emb, n_qs)
                rel_history.append(rel_summary)

        # Final scoring
        h_qs = q_emb[nodes[:, 0], :]
        scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        scores_all = scores_all.squeeze().cpu().data.numpy()

        # Get top-10 predictions
        n = 10
        top_indices = np.argsort(scores_all)[::-1][:n]
        answer = top_indices

        softscore = self.softmax(scores_all)
        probs = softscore[top_indices]

        # Write one question's top-k paths and close promptly so long exports
        # keep making visible progress on disk.
        with open(filepath, 'a') as f:
            # Use qid relative to test set start (same as DualR)
            qs = rels - self.loader.n_valid_qs - self.loader.n_train_qs
            f.write(f'{qs[0]}\t')

            # For each top-k prediction, trace back the path
            for k in range(n):
                tails = answer[k]
                f.write('%s|%0.3f|' % (self.loader.id2entity[answer[k]], probs[k]))

                # Trace path backwards from final answer
                print_edges = []
                for i in range(self.n_layer - 1, -1, -1):
                    edges = all_edges[i]
                    weights = all_weights[i]
                    mask1 = edges[:, 3] == tails
                    if np.sum(mask1) == 0:
                        tails = edges[0, 3]
                        mask1 = edges[:, 3] == tails
                    weights1 = weights[mask1].reshape(-1, 1)
                    edges1 = edges[mask1]
                    mask2 = np.argmax(weights1)

                    new_edges = edges1[mask2].reshape(1, -1)
                    new_weights = np.round_(weights1[mask2], 2).reshape(-1, 1)
                    new_edges = np.concatenate([new_edges[:, [1, 2, 3]], new_weights], 1)
                    tails = new_edges[:, 0].astype('int')
                    print_edges.append(new_edges)

                # Write path in forward order
                for i in range(self.n_layer - 1, -1, -1):
                    edge = print_edges[i][0].tolist()
                    if edge[1] < self.loader.n_rel:
                        h = self.loader.id2entity[int(edge[0])]
                        r = self.loader.id2relation[int(edge[1])]
                        t = self.loader.id2entity[int(edge[2])]
                        f.write('(' + h + ', ' + r + ', ' + t + ');')
                    elif edge[1] == 2 * self.n_rel:
                        h = self.loader.id2entity[int(edge[0])]
                        r = self.loader.id2relation[int(edge[1])]
                        t = self.loader.id2entity[int(edge[2])]
                        f.write('(' + h + ', ' + r + ', ' + t + ');')
                    else:
                        h = self.loader.id2entity[int(edge[0])]
                        r = self.loader.id2relation[int(edge[1]) - self.loader.n_rel]
                        t = self.loader.id2entity[int(edge[2])]
                        f.write('(' + t + ', ' + r + ', ' + h + ');')
                f.write('\t')
            f.write('\n')

        return True

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)
