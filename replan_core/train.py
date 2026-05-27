import os
import argparse
import torch
import numpy as np
from load_data import DataLoader
from base_model import BaseModel


parser = argparse.ArgumentParser(description="Parser for RePlan")
parser.add_argument('--dataset', type=str, default='webqsp')
parser.add_argument('--load', action='store_true')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--K', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--emb_dim', type=int, default=3584,
                    help='Embedding dimension (3584 for Qwen2.5-7B, 5120 for LLaMA-2-13B)')
parser.add_argument('--emb_dir', type=str, default='../embedding',
                    help='Directory containing question/relation embedding .npy files')
parser.add_argument('--plan_emb_dir', type=str, default='../embedding',
                    help='Directory containing plan embedding .npy files')
parser.add_argument('--fast', action='store_true',
                    help='Fast experiment mode: 1/4 train, 500 val, 15 epochs, batch=64')
parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
parser.add_argument('--n_batch', type=int, default=None, help='Override batch size')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (0 to disable)')
parser.add_argument('--test_only', action='store_true', help='Skip training, load checkpoint and run test only')
parser.add_argument('--gen_path', action='store_true', help='Generate reasoning paths (use with --test_only)')
parser.add_argument('--run_name', type=str, default=None, help='Run name suffix for checkpoint and perf files (e.g. "full", "fast"). Prevents overwriting between runs.')
parser.add_argument('--save_last', action='store_true', help='Also save the latest epoch checkpoint as <run_id>_last_model.pt')
parser.add_argument('--checkpoint_kind', type=str, default='best', choices=['best', 'last'],
                    help='Checkpoint to load in --test_only mode')
parser.add_argument('--no_pretrain', action='store_true', help='A1 ablation: skip loading pretrained plan generator')
parser.add_argument('--no_rel_feedback', action='store_true', help='A2 ablation: disable relation history feedback to plan generator')
parser.add_argument('--freeze_planner', action='store_true', help='A3 ablation: freeze plan generator parameters, only train GNN')
parser.add_argument('--projection_mode', type=str, default='residual',
                    choices=['residual', 'direct', 'static', 'layer_control', 'additive'],
                    help='Guidance variant: residual (main RePlan), direct (dynamic planner with shared/direct projection), static (fixed offline plan guidance baseline), layer_control (learned layer-wise residual guidance without plan supervision), or additive (residual-null: q + f_delta(p), no p-q subtraction)')

args = parser.parse_args()
if args.patience > 5:
    print(f'[INFO] Capping early-stopping patience from {args.patience} to 5')
    args.patience = 5


class Options(object):
    pass


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = args.dataset

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # run_name controls checkpoint and perf file naming to prevent overwriting
    run_suffix = ('_' + args.run_name) if args.run_name else ('_fast' if args.fast else '')
    run_id = dataset.replace('/', '-') + run_suffix
    ckpt_name = run_id + '_saved_model.pt'
    last_ckpt_name = run_id + '_last_model.pt'

    opts = Options
    opts.perf_file = os.path.join(results_dir, run_id + '_perf.txt')

    gpu = args.gpu
    torch.cuda.set_device(gpu)

    if dataset == 'MetaQA/1-hop':
        opts.lr = 0.00005
        opts.decay_rate = 0.996
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 1
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 40
        loaders = [DataLoader(args.dataset, plan_emb_dir=args.plan_emb_dir)]
    elif dataset == 'MetaQA/2-hop':
        opts.lr = 0.00004
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 2
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 40
        loaders = [DataLoader(args.dataset, plan_emb_dir=args.plan_emb_dir)]
    elif dataset == 'MetaQA/3-hop':
        opts.lr = 0.00004
        opts.decay_rate = 0.998
        opts.lamb = 0.00014
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 40
        loaders = [DataLoader(args.dataset, plan_emb_dir=args.plan_emb_dir)]
    elif dataset == 'webqsp':
        opts.lr = 0.00001
        opts.decay_rate = 0.9991
        opts.lamb = 0.00001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        opts.sample = 1
        loaders = [DataLoader(args.dataset, plan_emb_dir=args.plan_emb_dir)]
    elif dataset == 'CWQ':
        opts.lr = 0.00001
        opts.decay_rate = 0.993
        opts.lamb = 0.0001
        opts.hidden_dim = 256
        opts.attn_dim = 5
        opts.n_layer = 3
        opts.dropout = 0.1
        opts.act = 'idd'
        opts.n_batch = 20
        opts.n_tbatch = 20
        opts.K = 200
        opts.sample = 1
        loaders = [DataLoader(args.dataset, plan_emb_dir=args.plan_emb_dir)]

    opts.n_rel = loaders[0].n_rel
    opts.emb_dim = args.emb_dim
    opts.emb_dir = args.emb_dir
    opts.no_pretrain = args.no_pretrain
    opts.no_rel_feedback = args.no_rel_feedback
    opts.freeze_planner = args.freeze_planner
    opts.projection_mode = args.projection_mode

    # --fast mode overrides
    if args.fast:
        opts.n_batch = 64
        opts.n_tbatch = 64
    if args.n_batch is not None:
        opts.n_batch = args.n_batch
        opts.n_tbatch = args.n_batch
    n_epochs = args.epochs if args.epochs else (15 if args.fast else 40)

    # Fast mode: subsample training data to 1/4
    if args.fast:
        for loader in loaders:
            n_orig = len(loader.train_data)
            np.random.shuffle(loader.train_data)
            loader.train_data = loader.train_data[:n_orig // 4]
            loader.n_train = len(loader.train_data)
            # Subsample valid to 500 questions
            loader.n_valid = min(loader.n_valid, 500)
            print(f'[FAST] train: {n_orig} -> {loader.n_train}, valid: {loader.n_valid}')

    best_h1 = 0
    start_epoch = 0
    no_improve = 0
    for loader in loaders:
        model = BaseModel(opts, loader)

        # --test_only: load checkpoint and run test, skip training
        if args.test_only:
            ckpt_path = last_ckpt_name if args.checkpoint_kind == 'last' else ckpt_name
            if not os.path.exists(ckpt_path):
                print(f'Error: {ckpt_path} not found')
                exit(1)
            model.model.load_state_dict(torch.load(ckpt_path, map_location=f'cuda:{gpu}'))
            print(f'Loaded checkpoint from {ckpt_path}')

            if args.gen_path:
                # Generate reasoning paths for case study
                path_file = os.path.join(results_dir, dataset.replace('/', '-') + '-test-path.txt')
                model.get_path(mode='test', filepath=path_file)
                print(f'Paths generated: {path_file}')
            else:
                # Run test evaluation
                pred_file = os.path.join(results_dir, run_id + '_predictions.json')
                t_h1, test_str = model.evaluate_test(save_predictions=pred_file)
                print('Test: ' + test_str)
                with open(opts.perf_file, 'a') as f:
                    f.write('final_test:\n' + test_str)
            exit(0)

        # Resume from checkpoint if --load
        if args.load:
            ckpt_path = ckpt_name
            if os.path.exists(ckpt_path):
                model.model.load_state_dict(torch.load(ckpt_path, map_location=f'cuda:{gpu}'))
                print(f'Loaded checkpoint from {ckpt_path}')
                # Try to infer start_epoch from perf file
                if os.path.exists(opts.perf_file):
                    with open(opts.perf_file, 'r') as f:
                        lines = f.readlines()
                    best_epoch = None
                    for line in reversed(lines):
                        if line.strip() and line[0].isdigit() and '+' in line:
                            start_epoch = int(line.split()[0]) + 1
                            break
                    for line in lines:
                        if line.strip() and line[0].isdigit() and '+[VAL]' in line:
                            try:
                                h1 = float(line.split('H@1:')[1].split()[0])
                            except Exception:
                                continue
                            if h1 > best_h1:
                                best_h1 = h1
                                best_str = line.split('+', 1)[1]
                                best_epoch = int(line.split()[0])
                    if best_epoch is not None:
                        print(f'Resumed best validation H@1={best_h1:.4f} from epoch {best_epoch}')
                    print(f'Resuming from epoch {start_epoch}')
            else:
                print(f'Warning: --load specified but {ckpt_path} not found, training from scratch')

        perf_str = dataset + f' [projection={args.projection_mode}]' + '\n'
        # Load existing perf content if resuming
        if args.load and os.path.exists(opts.perf_file):
            with open(opts.perf_file, 'r') as f:
                perf_str = f.read()
                # Remove trailing best/final_test lines if present
                clean_lines = []
                for line in perf_str.split('\n'):
                    if line.startswith('best:') or line.startswith('final_test:') or line.startswith('[VAL]') or line.startswith('[TEST]'):
                        break
                    clean_lines.append(line)
                perf_str = '\n'.join(clean_lines)
                if not perf_str.endswith('\n'):
                    perf_str += '\n'

        for epoch in range(start_epoch, n_epochs):
            print('epoch:', epoch)
            v_h1, out_str, out_str2 = model.train_batch(evaluate=True)
            perf_str += str(epoch) + ' +' + out_str
            print(str(epoch) + ' +' + out_str)
            with open(opts.perf_file, 'w') as f:
                f.write(perf_str)
            if args.save_last:
                torch.save(model.model.state_dict(), last_ckpt_name)
            if v_h1 > best_h1:
                best_h1 = v_h1
                best_str = out_str
                no_improve = 0
                model.save_model(ckpt_name)
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print(f'Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
                    break
        perf_str += 'best:\n' + best_str
        with open(opts.perf_file, 'w') as f:
            f.write(perf_str)

        # Final test evaluation with best model
        print('Loading best model for final test evaluation...')
        model.model.load_state_dict(torch.load(ckpt_name, map_location=f'cuda:{gpu}'))
        t_h1, test_str = model.evaluate_test()
        print('Final test: ' + test_str)
        perf_str += 'final_test:\n' + test_str
        if args.save_last and os.path.exists(last_ckpt_name):
            print('Loading last model for final test evaluation...')
            model.model.load_state_dict(torch.load(last_ckpt_name, map_location=f'cuda:{gpu}'))
            last_h1, last_test_str = model.evaluate_test()
            print('Final last-epoch test: ' + last_test_str)
            perf_str += 'final_test_last:\n' + last_test_str
        with open(opts.perf_file, 'w') as f:
            f.write(perf_str)

        # Generate paths for Phase 2 (LLM answer selection)
        model.get_path(mode='test', filepath=os.path.join(results_dir, dataset.replace('/', '-') + '-test-path.txt'))
