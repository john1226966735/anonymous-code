import torch
import numpy as np
import time
import inspect
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import cal_ranks, cal_performance, cal_top1


def build_replan_model(args, loader):
    projection_mode = getattr(args, 'projection_mode', 'residual')
    if projection_mode == 'direct':
        from replan_model import RePlanExplore
    elif projection_mode == 'static':
        from models import PlanExplore as RePlanExplore
    elif projection_mode in ('residual', 'layer_control', 'additive'):
        from replan_model_v1 import RePlanExplore
    else:
        raise ValueError(f'Unknown projection_mode: {projection_mode}')
    print(f'Using projection_mode={projection_mode}')
    return RePlanExplore(args, loader)


class BaseModel(object):
    def __init__(self, args, loader):
        self.model = build_replan_model(args, loader)
        self.model.cuda()

        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_batch = args.n_batch
        self.n_tbatch = args.n_tbatch

        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.n_layer = args.n_layer
        self.args = args

        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ExponentialLR(self.optimizer, args.decay_rate)
        self.smooth = 1e-5
        self.t_time = 0

    def change_loader(self, loader):
        self.loader = loader
        self.n_ent = loader.n_ent
        self.n_rel = loader.n_rel
        self.n_train = loader.n_train
        self.n_valid = loader.n_valid
        self.n_test = loader.n_test
        self.model.change_loader(loader)

    def train_batch(self, evaluate=False):
        self.loader.shuffle_train()
        epoch_loss = 0
        i = 0

        batch_size = self.n_batch
        n_batch = self.loader.n_train // batch_size + (self.loader.n_train % batch_size > 0)
        print('n_batch:', n_batch)
        num_nodes = np.zeros((self.n_layer, 2))
        num_edges = np.zeros((self.n_layer, 2))
        t_time = time.time()
        self.model.train()

        if 'MetaQA/2-hop' in self.loader.task_dir or 'MetaQA/3-hop' in self.loader.task_dir:
            n_batch = n_batch // 10

        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min(self.loader.n_train, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx)

            self.model.zero_grad()
            n_nodes, n_edges, scores = self.model(subs, rels)

            num_nodes += n_nodes / n_batch
            num_edges += n_edges / n_batch

            pos_scores = scores[[torch.arange(len(scores)).cuda(), torch.LongTensor(objs).cuda()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))

            loss.backward()
            self.optimizer.step()

            # avoid NaN
            for p in self.model.parameters():
                X = p.data.clone()
                flag = X != X
                X[flag] = np.random.random()
                p.data.copy_(X)
            epoch_loss += loss.item()

        self.scheduler.step()
        self.t_time += time.time() - t_time
        print('epoch_loss:', epoch_loss, 'time:', self.t_time)
        out_str2 = str(num_nodes.reshape(1, -1).astype(int)) + '\n' + str(num_edges.reshape(1, -1).astype(int)) + '\n'

        if evaluate:
            v_h1, out_str = self.evaluate()
            return v_h1, out_str, out_str2
        else:
            return 0, '', out_str2

    def evaluate(self):
        """Evaluate on validation set only. Returns v_h1 for model selection."""
        batch_size = self.n_tbatch
        print('valid:')
        n_data = self.n_valid
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        self.model.eval()
        i_time = time.time()
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='valid')
            scores = self.model(subs, rels, mode='valid').data.cpu().numpy()
            filters = 0
            ranks = cal_ranks(scores, objs, filters)
            ranking += ranks
        ranking = np.array(ranking)
        v_mrr, v_h1, v_h3, v_h10 = cal_performance(ranking)
        i_time = time.time() - i_time

        out_str = '[VAL] H@1:%.4f H@10:%.4f\t[TIME] train:%.4f inference:%.4f\n' % (
            v_h1, v_h10, self.t_time, i_time)

        return v_h1, out_str

    def evaluate_test(self, save_predictions=None):
        """Evaluate on test set. Called once after training with best model.
        If save_predictions is a filepath, save per-question predictions to JSON.
        """
        batch_size = self.n_tbatch
        print('test:')
        n_data = self.n_test
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        ranking = []
        predictions = [] if save_predictions else None
        self.model.eval()
        i_time = time.time()
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data='test')
            scores = self.model(subs, rels, mode='test').data.cpu().numpy()
            filters = 0
            ranks = cal_top1(scores, objs, filters)
            ranking += ranks

            if predictions is not None:
                top1 = np.argmax(scores, axis=1)
                for j in range(len(batch_idx)):
                    predictions.append({
                        'idx': int(batch_idx[j]),
                        'rank': int(ranks[j]),
                        'correct': int(ranks[j] == 1)
                    })

        ranking = np.array(ranking)
        t_mrr, t_h1, _, _ = cal_performance(ranking)
        assert len(ranking) == self.n_test
        i_time = time.time() - i_time

        if save_predictions:
            import json
            with open(save_predictions, 'w') as f:
                json.dump(predictions, f)
            print(f'Predictions saved to {save_predictions} ({len(predictions)} items)')

        out_str = '[TEST] H@1:%.4f\t[TIME] inference:%.4f\n' % (t_h1, i_time)
        return t_h1, out_str

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def get_path(self, mode='test', filepath='path.txt'):
        print('generate path:')
        batch_size = 1
        if mode == 'test':
            n_data = self.n_test
        else:
            n_data = 500
        n = 10
        n_batch = n_data // batch_size + (n_data % batch_size > 0)
        f = open(filepath, 'w')
        self.model.eval()
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min(n_data, (i + 1) * batch_size)
            batch_idx = np.arange(start, end)
            subs, rels, objs = self.loader.get_batch(batch_idx, data=mode)
            visual_path_sig = inspect.signature(self.model.visual_path)
            if 'batch_idx' in visual_path_sig.parameters:
                self.model.visual_path(subs, rels, objs, batch_idx, filepath=filepath, mode=mode)
            else:
                self.model.visual_path(subs, rels, objs, filepath=filepath, mode=mode)

        print('path generate done in ' + filepath + '\n')

    def save_model(self, ckpt_path='', out_str=''):
        torch.save(self.model.state_dict(), ckpt_path)
        print('model saved to ' + ckpt_path)
