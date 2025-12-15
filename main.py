import numpy as np
import torch
import argparse
import os
import time
import torch.nn as nn
import torch.nn.functional as F
from datasets import HyperX, get_dataset
from utils_HSI import sample_gt, metrics, seed_worker, AvgrageMeter
from datetime import datetime
from model import GFNetPyramid, DSPLTnet, NotearsClassifier
from functools import partial

from ptflops import get_model_complexity_info


parser = argparse.ArgumentParser(description='PyTorch: Harnessing Spectral Low-Frequency Stability and Causal Invariance for Cross-Scene Hyperspectral Image Classification ')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the test dir')
parser.add_argument('--patch_size', type=int, default=8)

parser.add_argument('--lr', default=2.5e-4, type=float)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--temperature', type=float, default=0.07)

parser.add_argument('--batch_size', default=256, type=int) 

parser.add_argument('--training_sample_ratio', type=float, default=1,
                    help='training sample ratio')

parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--seed', type=int, default=333,
                    help='random seed ')

parser.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
parser.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
parser.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

parser.add_argument('--log_path', type=str, default='./logs',
                    help='the path to load the tensorboard data')

# Spectral Patch Low-Frequency Transformation Network parameters
parser.add_argument('--perturb_prob', default=0.5, type=float)
parser.add_argument('--mask_alpha', default=0.2, type=float)

parser.add_argument('--noise_mode', default=1, type=int, help="0: close; 1: add noise")
parser.add_argument('--uncertainty_factor', default=1.0, type=float)
parser.add_argument('--mask_radio', default=0.1, type=float) 
parser.add_argument('--gauss_or_uniform', default=0, type=int, help="0: gaussian; 1: uniform; 2: random")
parser.add_argument('--noise_layers', default=[0, 1], nargs="+", type=int, help="where to use augmentation.")

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# DAG parameters
parser.add_argument('--out_dim', default=256, type=int)
parser.add_argument('--hidden_size', default=256, type=int)
parser.add_argument('--num_hidden_layers', default=0, type=int)
parser.add_argument('--ema_ratio', default=0.99, type=float)
parser.add_argument('--lambda1', default=1, type=float)
parser.add_argument('--lambda2', default=1, type=float)
parser.add_argument('--rho_max', default=100, type=float)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--rho', default=1.0, type=float)
parser.add_argument('--weight_dag', default=0.1, type=float)
parser.add_argument('--weight_con', default=0.1, type=float)
parser.add_argument('--dag_anneal_steps', default=200, type=int)

# Optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay (default: 0.05)')
parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')

args = parser.parse_args()

seed_worker(args.seed)


def evaluate(net, val_loader, gpu):
    ps = []
    ys = []
    loss_val_metric = AvgrageMeter()
    with torch.no_grad():

        for i,(x1, y1) in enumerate(val_loader):
            x1 = x1.to(gpu)
            y1 = y1.to(gpu)
            y1 = y1 - 1
            p1 = net(x1)
            loss_val = nn.CrossEntropyLoss()(p1,y1).item()
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.cpu().numpy())
            loss_val_metric.update(loss_val, x1.shape[0])

        ps = np.concatenate(ps)
        ys = np.concatenate(ys)
        acc = np.mean(ys==ps)*100
        results = metrics(ps, ys, n_classes=ys.max() + 1)
    return acc, results, loss_val_metric.get_avg()

def main():

    train_res = {
        'best_epoch': 0,
        'best_acc': 0,
        'Confusion_matrix': [],
        'OA': 0,
        'TPR': 0,
        'F1scores': 0,
        'kappa': 0
    }

    hyperparams = vars(args)
    s = ''
    for k, v in args.__dict__.items():
        s += '\t' + k + '\t' + str(v) + '\n'

    f = open(log_dir + '/settings.txt', 'w+')
    f.write(s)
    f.close()

    img_src, gt_src, _, IGNORED_LABELS, _, _ = get_dataset(args.source_name, args.data_path)
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    img_tar, gt_tar, _, IGNORED_LABELS, _, _ = get_dataset(args.target_name, args.data_path)

    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))

    train_gt_src, _, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')

    img_src_con, train_gt_src_con = img_src, train_gt_src

    hyperparams_train = hyperparams.copy()
    hyperparams_train['flip_augmentation'] = True
    hyperparams_train['radiation_augmentation'] = True

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=hyperparams['batch_size'],
                                         pin_memory=True,
                                         worker_init_fn=seed_worker,
                                         generator=g,
                                         shuffle=True)

    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                         pin_memory=True,
                                         batch_size=hyperparams['batch_size'])


    BLT = GFNetPyramid(
            img_size=hyperparams['patch_size'],
            patch_size=4,
            in_chans = hyperparams['n_bands'],
            num_classes=args.n_classes,
            embed_dim=[128, 256], depth=[2, 2],
            mlp_ratio=[2,2],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1,
            mask_radio=args.mask_radio,
            mask_alpha=args.mask_alpha,
            noise_mode=args.noise_mode,
            perturb_prob=args.perturb_prob,
            noise_layers=args.noise_layers, gauss_or_uniform=args.gauss_or_uniform,
        )

    model = DSPLTnet(model=BLT, NotearsClassifier=NotearsClassifier, num_classes=num_classes, hparams=hyperparams)

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params / 1e6:.2f}M trainable parameters.')

    # --- 计算 FLOPS using ptflops ---
    model.cpu()
    # model.eval()  # Set the model to evaluation mode for FLOPs calculation
    try:
        sample_x, _ = next(iter(train_loader))
        input_shape = tuple(sample_x.shape[1:])
    
        print(f"Sample input shape for FLOPs calculation: {input_shape}")
        
        macs, params = get_model_complexity_info(model, input_shape, as_strings=True,
                                                print_per_layer_stat=False, verbose=False)
        
        print(f"Total FLOPs: {macs}") 
        print(f"Total Parameters: {params}")
        
    except StopIteration:
        print("Warning: train_loader is empty, cannot calculate FLOPs with a sample input.")
    # --- FLOPs 计算结束 ---

    model.to(args.gpu) 

    loss_metric = AvgrageMeter()

    best_acc = 0
    step=0
    for epoch in range(1, args.epochs + 1):
        loss_metric.reset()
        model.train()
        t1 = time.time()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            step = step + 1
            loss_dict = model.update(x, y, step=step)
            loss = loss_dict["loss"]
            loss_metric.update(loss, x.shape[0])


        t2 = time.time()

        print("[TRAIN EPOCH {}] loss={} Time={:.2f}".format(epoch, loss_metric.get_avg(), t2 - t1))

        model.eval()
        taracc, results, _ = evaluate(model, test_loader, args.gpu)

        if best_acc < taracc:
            best_acc = taracc
            torch.save({'model':model.state_dict()}, os.path.join(log_dir, f'source_classifier.pkl'))
            train_res['best_epoch'] = epoch
            train_res['best_acc'] = '{:.2f}'.format(best_acc)
            train_res['Confusion_matrix'] = '{:}'.format(results['Confusion_matrix'])
            train_res['OA'] = '{:.2f}'.format(results['Accuracy'])
            train_res['TPR'] = '{:}'.format(np.round(results['TPR'] * 100, 2))
            train_res['F1scores'] = '{:}'.format(results["F1_scores"])
            train_res['kappa'] = '{:.4f}'.format(results["Kappa"])

        print(f'[TRAIN EPOCH {epoch}] taracc {taracc:2.2f} best_taracc {best_acc:2.2f}')

    with open(log_dir + '/train_log.txt', 'w+') as f:
        for key, value in train_res.items():
            f.write(f"{key}: {value}\n")
    f.close()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y%m%d%H%M%S')
    exp_name = '{}/{}'.format(args.save_path, args.source_name+'to'+args.target_name+'_'+ time_str)

    timestamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_dir = os.path.join(BASE_DIR, exp_name, 'lr_'+ str(args.lr)+
                             '_pt'+str(args.patch_size) + '_batch_size' + str(args.batch_size) + '_dag_anneal_steps' + str(args.dag_anneal_steps) + '_weight_dag' + str(args.weight_dag) + '_weight_con' + str(args.weight_con) + '_' +timestamp)
    log_dir = log_dir.replace('\\', '/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    main()