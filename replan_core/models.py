import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np


def variadic_topk(values, counts, k):
    """Pure-PyTorch replacement for torchdrug.layers.functional.variadic_topk.

    For each group (defined by counts), return the top-k values and their
    indices within the flat `values` tensor.

    Args:
        values: 1-D tensor of scores
        counts: 1-D tensor, number of elements per group
        k: int, how many to keep per group
    Returns:
        topk_values: (n_groups, k) padded with -inf
        topk_indices: (n_groups, k) padded with 0
    """
    n_groups = counts.size(0)
    device = values.device
    topk_values = torch.full((n_groups, k), float('-inf'), device=device)
    topk_indices = torch.zeros((n_groups, k), dtype=torch.long, device=device)

    offset = 0
    for i in range(n_groups):
        c = counts[i].item()
        group_vals = values[offset : offset + c]
        actual_k = min(k, c)
        if actual_k > 0:
            tk_vals, tk_idx = torch.topk(group_vals, actual_k)
            topk_values[i, :actual_k] = tk_vals
            topk_indices[i, :actual_k] = tk_idx  # indices relative to group start
        offset += c

    return topk_values, topk_indices


class GNNLayer(torch.nn.Module):
    """GNN layer with attention-based message passing.

    Unchanged from DualR — the caller controls what "guide_emb" is passed
    (question embedding in DualR, plan step embedding in PlanR).
    """
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, use_lama_rel, K, sample_flag, act=lambda x: x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.use_lama_rel = use_lama_rel
        self.K = K
        self.sample_flag = sample_flag

        self.Ws_attn = nn.Linear(in_dim, attn_dim)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wq_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.w_alpha = nn.Linear(attn_dim, 1)

        self.W_h = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, q_sub, q_rel, guide_emb, rela_embed, hidden, edges, nodes, old_nodes_new_idx):
        """
        Args:
            guide_emb: (batch_size, hidden_dim) — the attention guidance signal.
                       In DualR this is always q_emb (question embedding).
                       In PlanR this is plan_step_emb for the current layer.
        """
        l1 = edges.shape[0]
        n1 = nodes.size(0)
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        if self.use_lama_rel == 1:
            hr = rela_embed[rel, :]
        else:
            hr = rela_embed(rel)

        self.n_rel = (rela_embed.shape[0] - 1) // 2

        r_idx = edges[:, 0]
        h_guide = guide_emb[edges[:, 0], :]

        # First pass: attention for top-K sampling
        message = hs + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
            self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_guide) + self.Wqr_attn(hr * h_guide)
        )))

        sample_flag = self.sample_flag
        if sample_flag == 1:
            max_ent_per_ent = self.K
            _, ind1 = torch.unique(edges[:, 0:2], dim=0, sorted=True, return_inverse=True)
            _, ind2 = torch.sort(ind1)
            edges = edges[ind2]
            alpha = alpha[ind2]
            _, counts = torch.unique(edges[:, 0:2], dim=0, return_counts=True)

            idd_idx = edges[:, 2] == (self.n_rel * 2)
            idd_edges = edges[idd_idx]

            probs = alpha.squeeze()
            topk_value, topk_index = variadic_topk(probs, counts, k=max_ent_per_ent)

            cnt_sum = torch.cumsum(counts, dim=0)
            cnt_sum[1:] = cnt_sum[:-1] + 0
            cnt_sum[0] = 0
            topk_index = topk_index + cnt_sum.unsqueeze(1)

            mask = topk_index.view(-1, 1).squeeze()
            mask = torch.unique(mask)

            edges = edges[mask]
            edges = torch.cat((edges, idd_edges), 0)
            edges = torch.unique(edges[:, :], dim=0)

            nodes, tail_index = torch.unique(edges[:, [0, 3]], dim=0, sorted=True, return_inverse=True)
            edges = torch.cat([edges[:, 0:5], tail_index.unsqueeze(1)], 1)

            head_index = edges[:, 4]
            idd_mask = edges[:, 2] == (self.n_rel * 2)
            _, old_idx = head_index[idd_mask].sort()
            old_nodes_new_idx = tail_index[idd_mask][old_idx]
        else:
            pass

        # Second pass: message passing with attention
        sub = edges[:, 4]
        rel = edges[:, 2]
        obj = edges[:, 5]

        hs = hidden[sub]
        if self.use_lama_rel == 1:
            hr = rela_embed[rel, :]
        else:
            hr = rela_embed(rel)

        r_idx = edges[:, 0]
        h_guide = guide_emb[edges[:, 0], :]

        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(
            self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wq_attn(h_guide) + self.Wqr_attn(hr * h_guide)
        )))
        message = alpha * message

        message_agg = scatter(message, index=obj, dim=0, dim_size=nodes.size(0), reduce='sum')

        hidden_new = self.act(self.W_h(message_agg))

        l2 = edges.shape[0]
        n2 = nodes.size(0)
        num_node = np.array([n1 * 1.0 / len(q_sub), n2 * 1.0 / len(q_sub)])
        num_edge = np.array([l1 * 1.0 / len(q_sub), l2 * 1.0 / len(q_sub)])

        return num_node, num_edge, hidden_new, alpha, nodes, edges, old_nodes_new_idx


class PlanExplore(torch.nn.Module):
    """PlanR model: Plan-guided GNN exploration.

    Key difference from DualR's Explore:
    - Each GNN layer receives a different plan step embedding as the attention
      guidance signal, instead of the same question embedding for all layers.
    - Node initialization and final scoring still use question embedding.
    - Plan embeddings share dim_reduct with question embeddings (same MLP),
      so when plan degrades to question, PlanR degrades to DualR exactly.
    """
    def __init__(self, params, loader):
        super(PlanExplore, self).__init__()
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x: x}
        act = acts[params.act]
        self.K = params.K
        self.sample_flag = params.sample

        self.emb_dim = getattr(params, 'emb_dim', 5120)
        self.emb_dir = getattr(params, 'emb_dir', '../embedding')

        # Question embeddings + dim reduction (same as DualR)
        self.question_emb = self.load_qemb().detach()
        mid_dim = min(2096, self.emb_dim)
        self.dim_reduct = nn.Sequential(
            nn.Linear(self.emb_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, self.hidden_dim)
        )

        # Relation embeddings (same as DualR)
        self.use_lama_rel = 1
        if self.use_lama_rel == 1:
            self.rela_embed = self.load_rel_emb().detach()
        else:
            self.rela_embed = nn.Embedding(2 * self.n_rel + 1, self.hidden_dim)

        # ---- PlanR addition: plan embeddings ----
        self.plan_emb = None
        if loader.plan_emb is not None:
            self.plan_emb = torch.tensor(loader.plan_emb, dtype=torch.float32)
            print(f'Plan embeddings loaded into model: {self.plan_emb.shape}')

        # ---- v1: independent MLP for plan-question residual projection ----
        self.plan_delta_proj = nn.Sequential(
            nn.Linear(self.emb_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, self.hidden_dim)
        )

        # GNN layers (same as DualR)
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

    def _get_plan_guide(self, q_id, q_emb, layer_idx):
        """Get the guidance embedding for a specific GNN layer.

        v1: guide = q_emb + plan_delta_proj(plan_raw - question_raw)
        When plan_step = question (or zero), delta = 0, guide ≈ q_emb (degrades to DualR).
        """
        if self.plan_emb is not None:
            max_steps = self.plan_emb.size(1)
            if layer_idx < max_steps:
                plan_raw = self.plan_emb[q_id, layer_idx, :]  # (batch, emb_dim) CPU
                nonzero_mask = plan_raw.abs().sum(dim=-1) > 0  # (batch,) CPU
                if nonzero_mask.any():
                    plan_raw_gpu = plan_raw.cuda()
                    q_raw_gpu = self.question_emb[q_id, :].cuda()  # (batch, emb_dim)
                    delta = plan_raw_gpu - q_raw_gpu                # (batch, emb_dim)
                    delta_proj = self.plan_delta_proj(delta)        # (batch, hidden_dim)

                    del plan_raw_gpu, q_raw_gpu, delta

                    nonzero_mask_gpu = nonzero_mask.cuda()
                    guide = q_emb.clone()
                    guide[nonzero_mask_gpu] = q_emb[nonzero_mask_gpu] + delta_proj[nonzero_mask_gpu]
                    return guide
        return q_emb

    def forward(self, subs, qids, mode='train'):
        n_qs = len(qids)
        q_sub = subs
        q_id = torch.LongTensor(qids)

        # Question embedding (for node init + final scoring)
        ques_emb = self.question_emb[q_id, :]
        ques_emb = ques_emb.cuda()
        q_id = q_id.cuda()
        q_emb = self.dim_reduct(ques_emb)
        ques_emb.cpu()

        # Relation embedding (same as DualR)
        if self.use_lama_rel == 1:
            self.rela_embed = self.rela_embed.cuda()
            rel_emb = self.dim_reduct(self.rela_embed)
            self.rela_embed.cpu()
            rel_emb = rel_emb[0:self.n_rel, :]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop], dim=0)
        else:
            rel_emb = self.rela_embed

        # Node initialization (same as DualR: use question embedding)
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

        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))
        scores_all = []

        for i in range(self.n_layer):
            # ---- PlanR core change: get plan step embedding for this layer ----
            guide_emb = self._get_plan_guide(torch.LongTensor(qids), q_emb, layer_idx=i)

            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids)
            num_node, num_edge, hidden, alpha, nodes, edges, old_nodes_new_idx = \
                self.gnn_layers[i](q_sub, q_id, guide_emb, rel_emb, hidden, edges, nodes, old_nodes_new_idx)

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            num_nodes[i, :] += num_node
            num_edges[i, :] += num_edge

        # Final scoring (same as DualR: use question embedding)
        h_qs = q_emb[nodes[:, 0], :]
        scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores

        if mode == 'train':
            return num_nodes, num_edges, scores_all
        else:
            return scores_all

    def load_qemb(self):
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

    def change_loader(self, loader):
        self.loader = loader
        self.question_emb = self.load_qemb().detach()
        self.rela_embed = self.load_rel_emb().detach()
        self.n_rel = self.loader.n_rel
        if loader.plan_emb is not None:
            self.plan_emb = torch.tensor(loader.plan_emb, dtype=torch.float32)
        print('change loader:', self.loader.task_dir)

    def visual_path(self, subs, qids, objs, filepath, mode='test'):
        n_qs = len(qids)
        q_sub = subs
        q_id = torch.LongTensor(qids)

        ques_emb = self.question_emb[q_id, :]
        ques_emb = ques_emb.cuda()
        q_id = q_id.cuda()
        q_emb = self.dim_reduct(ques_emb)
        ques_emb.cpu()

        if self.use_lama_rel == 1:
            self.rela_embed = self.rela_embed.cuda()
            rel_emb = self.dim_reduct(self.rela_embed)
            self.rela_embed.cpu()
            rel_emb = rel_emb[0:self.n_rel, :]
            rev_rel_emb = self.Wr(rel_emb)
            rel_emb = torch.concat([rel_emb, rev_rel_emb, self.loop], dim=0)
        else:
            rel_emb = self.rela_embed

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

        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))

        all_nodes = []
        all_edges = []
        all_weights = []
        min_weight = []

        for i in range(self.n_layer):
            # ---- PlanR: use plan step embedding for this layer ----
            guide_emb = self._get_plan_guide(torch.LongTensor(qids), q_emb, layer_idx=i)

            nodes, edges, old_nodes_new_idx = self.loader.get_neighbors(nodes.data.cpu().numpy(), qids)
            num_node, num_edge, hidden, weights, nodes, edges, old_nodes_new_idx = \
                self.gnn_layers[i](q_sub, q_id, guide_emb, rel_emb, hidden, edges, nodes, old_nodes_new_idx)

            h0 = torch.zeros(1, nodes.size(0), hidden.size(1)).cuda().index_copy_(1, old_nodes_new_idx, h0)
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)

            all_nodes.append(nodes.cpu().data.numpy())
            all_edges.append(edges.cpu().data.numpy())
            all_weights.append(weights.cpu().data.numpy())
            min_weight.append(torch.min(weights).item())

        h_qs = q_emb[nodes[:, 0], :]
        scores = self.mlp(torch.cat((hidden, h_qs), dim=1)).squeeze(-1)
        scores_all = torch.zeros((n_qs, self.loader.n_ent)).cuda()
        scores_all[[nodes[:, 0], nodes[:, 1]]] = scores
        scores_all = scores_all.squeeze().cpu().data.numpy()
        n = 10
        top_indices = np.argsort(scores_all)[::-1][:n]
        answer = top_indices

        softscore = self.softmax(scores_all)
        top_values = np.partition(softscore, -2)[::-1][:n]
        probs = top_values

        f = open(filepath, '+a')
        qs = qids - self.loader.n_valid_qs - self.loader.n_train_qs

        f.write(f'{qs[0]}\t')

        for k in range(n):
            tails = answer[k]
            outstr = 'tail: %d,  p:%.2f' % (tails, probs[k])

            f.write('%s|%0.3f|' % (self.loader.id2entity[answer[k]], probs[k]))
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

            for i in range(self.n_layer - 1, -1, -1):
                edge = print_edges[i][0].tolist()
                outstr = '%d\t %d\t %d\t%.4f' % (edge[0], edge[1], edge[2], edge[3])

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
