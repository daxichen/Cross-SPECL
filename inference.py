import numpy as np
import torch
import argparse
import os
import scipy.io as sio
from datasets import HyperX_test, get_dataset
from utils_HSI import seed_worker, test_metrics

from model import GFNetPyramid, DSPLTnet, NotearsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial
import torch.nn as nn

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='PyTorch: Harnessing Spectral Low-Frequency Stability and Causal Invariance for Cross-Scene Hyperspectral Image Classification ')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the source dir')
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2.5e-4,
                    help="Learning rate, set by the model if not specified.")

parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")

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

parser.add_argument('--log_path', type=str, default='./logs',
                    help='the path to load the tensorboard data')

# Spectral Patch Low-Frequency Transformation Network parameters
parser.add_argument('--perturb_prob', default=0.5,  type=float)
parser.add_argument('--mask_alpha', default=0.2,  type=float)

parser.add_argument('--noise_mode', default=1,  type=int, help="0: close; 1: add noise")
parser.add_argument('--uncertainty_factor', default=1.0,  type=float)
parser.add_argument('--mask_radio', default=0.1,  type=float) # base=0.1
parser.add_argument('--gauss_or_uniform', default=0,  type=int, help="0: gaussian; 1: uniform; 2: random")
parser.add_argument('--noise_layers', default=[0, 1],  nargs="+", type=int, help="where to use augmentation.")
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# DAG parameters
parser.add_argument('--out_dim', default=256,  type=int)
parser.add_argument('--hidden_size', default=256,  type=int)
parser.add_argument('--num_hidden_layers', default=0,  type=int)
parser.add_argument('--ema_ratio', default=0.99,  type=float)
parser.add_argument('--lambda1', default=1,  type=float)
parser.add_argument('--lambda2', default=1,  type=float)
parser.add_argument('--rho_max', default=100,  type=float)
parser.add_argument('--alpha', default=1.0,  type=float)
parser.add_argument('--rho', default=1.0,  type=float)
parser.add_argument('--weight_dag', default=0.1,  type=float)
parser.add_argument('--weight_con', default=0.1,  type=float)
parser.add_argument('--dag_anneal_steps', default=200,  type=int)

# Optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay (default: 0.05)')
parser.add_argument('--optimizer', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    result_record = {
        'Confusion_matrix': [],
        'OA': 0,
        'TPR': 0,
        'F1scores': 0,
        'kappa': 0
    }


    root = os.path.join(args.save_path, args.target_name +'_results')
    log_dir = os.path.join(root, 'lr_'+ str(args.lr)+
                           '_pt'+str(args.patch_size))
    
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_worker(args.seed) 

    img_tar, gt_tar, _, IGNORED_LABELS, _, _ = get_dataset(args.target_name, args.data_path)
    hyperparams = vars(args)
    num_classes = gt_tar.max()
    N_BANDS = img_tar.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})
    r = int(hyperparams['patch_size']/2)+1
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))

    hyperparams_train = hyperparams.copy()
    hyperparams_train['radiation_augmentation'] = True

    test_dataset = HyperX_test(img_tar, gt_tar, **hyperparams)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    shuffle=False)
    
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
    
    model.to(args.gpu)


    model_path = 'results\Houston13toHouston18_20240606222003\lr_0.00025_pt8_batch_size256_dag_anneal_steps200_weight_dag0.1_weight_con0.1_202406062220\source_classifier.pkl'
    save_weight = torch.load(model_path,map_location='cuda:0' )
    model.load_state_dict(save_weight['model'], strict=False)


    model = model.to(args.gpu)

    model.eval()
    results = []
    Row, Column = [], []


    for i, (x, y, center_coords) in tqdm(enumerate(test_loader), total=len(test_loader)):
    
        x, y = x.to(args.gpu), y.to(args.gpu)

        with torch.no_grad():

            
           
            pred_x = model(x)

            result = np.argmax(pred_x.cpu().numpy(), axis=-1) + 1
            results.extend(result)

            Row.extend([coord for coord in center_coords[0]])
            Column.extend([coord for coord in center_coords[1]])
    
    
    size = gt_tar.shape

    prediction = np.zeros((size[0],size[1]))
    for i, pred_label in enumerate(results):
        center_x, center_y = Row[i], Column[i]
        prediction[center_x, center_y] = pred_label


    prediction = prediction[r:-r, r:-r]

    path_prediction = log_dir + '/' + args.target_name + '.mat'
    sio.savemat(path_prediction, {'prediction': prediction})
    gt = gt_tar[r:-r, r:-r]

    plt.imshow(prediction, cmap='jet')
    plt.axis('off')
    plt.savefig('./results/' + args.target_name + '_predection' + '.png', bbox_inches = 'tight', pad_inches = 0, dpi = 600)
    plt.show()

    plt.imshow(gt, cmap='jet')
    plt.axis('off')
    plt.savefig('./results/' + args.target_name + '_gt' + '.png', bbox_inches = 'tight', pad_inches = 0, dpi = 600)
    plt.show()


    pred = prediction.reshape(-1)

    gt = gt.reshape(-1)
    results = test_metrics(
            pred,
            gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=num_classes,
        )
    print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
                results['Accuracy'], '\n', 'kappa:', results["Kappa"])
    
    result_record['Confusion_matrix']= '{:}'.format(results['Confusion_matrix'])
    result_record['OA'] = '{:.2f}'.format(results['Accuracy'])
    result_record['TPR'] = '{:}'.format(np.round(results['TPR'] * 100, 2))
    result_record['F1scores'] = '{:}'.format(results["F1_scores"])
    result_record['kappa'] = '{:.4f}'.format(results["Kappa"])

    with open(log_dir + '/results.txt', 'w+') as f:
        for key, value in result_record.items():
            f.write(f"{key}: {value}\n")
    f.close()


if __name__ == '__main__':
    main()