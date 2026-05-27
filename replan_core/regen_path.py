"""Regenerate test path file from a saved RePlan v1 checkpoint."""
import argparse
import os
import torch

from load_data import DataLoader
from base_model_v1 import BaseModel


def build_opts(args, loader):
    class Opts:
        pass

    opts = Opts()
    opts.gpu = args.gpu
    opts.K = 200
    opts.n_batch = 20
    opts.n_tbatch = 20
    opts.emb_dim = args.emb_dim
    opts.emb_dir = args.emb_dir
    opts.lr = 1e-5
    if args.dataset == 'webqsp':
        opts.lamb = 1e-5
        opts.decay_rate = 0.9991
    else:
        opts.lamb = 1e-4
        opts.decay_rate = 0.993
    opts.hidden_dim = 256
    opts.attn_dim = 5
    opts.n_layer = 3
    opts.dropout = 0.1
    opts.act = 'idd'
    opts.sample = 1
    opts.n_rel = loader.n_rel
    opts.plan_generator_ckpt = args.planner_ckpt
    return opts


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='webqsp')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--emb_dim', type=int, default=3584)
parser.add_argument('--emb_dir', type=str, default='../embedding')
parser.add_argument('--planner_ckpt', type=str, default=None,
                    help='Planner checkpoint used by this run')
parser.add_argument('--ckpt', type=str, default=None,
                    help='Model checkpoint; defaults to <dataset>_saved_model_v1.pt')
parser.add_argument('--path_file', type=str, default=None,
                    help='Output path file; defaults to results/<dataset>-test-path.txt')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.cuda.set_device(args.gpu)

loader = DataLoader(args.dataset, plan_emb_dir=None)
opts = build_opts(args, loader)
model = BaseModel(opts, loader)

ckpt = args.ckpt or f'{args.dataset}_saved_model_v1.pt'
state_dict = torch.load(ckpt, map_location='cpu')
model.model.load_state_dict(state_dict)
print(f'Loaded {ckpt}')

path_file = args.path_file or f'results/{args.dataset}-test-path.txt'
print(f'Regenerating path file: {path_file}')
model.get_path(mode='test', filepath=path_file)
print('Done.')
