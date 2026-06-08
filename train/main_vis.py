# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import os
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Imports for visualization
from PIL import Image
import matplotlib.pyplot as plt
from openTSNE import TSNE
import torchvision

from TCMM import datasets, models
from TCMM.models.cm import ClusterMemory
from TCMM.trainers import Trainer
from TCMM.evaluators import Evaluator, extract_features
from TCMM.utils.data import IterLoader
from TCMM.utils.data import transforms as T
from TCMM.utils.data.sampler import RandomMultipleGallerySampler
from TCMM.utils.data.preprocessor import Preprocessor
from TCMM.utils.logging import Logger
from TCMM.utils.serialization import load_checkpoint, save_checkpoint
from TCMM.utils.faiss_rerank import compute_jaccard_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_epoch = best_mAP = 0

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Visualization Functions Start
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Helper for t-SNE plot (from examples/utils.py)
MOUSE_10X_COLORS = {
    -1: "#000000", 0: "#FFFF00", 1: "#1CE6FF", 2: "#FF34FF", 3: "#FF4A46",
    4: "#008941", 5: "#006FA6", 6: "#A30059", 7: "#FFDBE5", 8: "#7A4900",
    9: "#0000A6", 10: "#63FFAC", 11: "#B79762", 12: "#004D43", 13: "#8FB0FF",
    14: "#997D87", 15: "#5A0007", 16: "#809693", 17: "#FEFFE6", 18: "#1B4400",
    19: "#4FC601", 20: "#3B5DFF", 21: "#4A3B53", 22: "#FF2F80", 23: "#61615A",
    24: "#BA0900", 25: "#6B7900", 26: "#00C2A0", 27: "#FFAA92", 28: "#FF90C9",
    29: "#B903AA", 30: "#D16100", 31: "#DDEFFF", 32: "#000035", 33: "#7B4F4B",
    34: "#A1C299", 35: "#300018", 36: "#0AA6D8", 37: "#013349", 38: "#00846F",
}

def plot_tsne(x, y, colors=None, **kwargs):
    import matplotlib
    _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))
    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}
    
    if colors is None:
        colors = MOUSE_10X_COLORS

    # Create a color for each unique pid to avoid None values
    unique_pids = np.unique(y)
    pid_to_color = {}
    # Use color keys 0 and above for cycling
    color_keys = sorted([k for k in colors.keys() if k != -1])
    num_colors = len(color_keys)

    for pid in unique_pids:
        if pid == -1:
            pid_to_color[pid] = colors.get(-1, "#000000")  # Default for noise
        else:
            pid_to_color[pid] = colors[color_keys[pid % num_colors]]
            
    point_colors = [pid_to_color[pid] for pid in y]

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")


def visualize_tsne(model, test_loader, args):
    print("==> Starting t-SNE visualization...")
    # Extract features for the entire test set
    features, _ = extract_features(model, test_loader)
    features_list = torch.stack(list(features.values()))
    
    # Get ground truth labels
    pids = []
    for _, pid, _ in test_loader.dataset.dataset:
        pids.append(pid)
    
    print(f"Running t-SNE on {len(features_list)} features...")
    tsne = TSNE(
        perplexity=30,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    embedding = tsne.fit(features_list.numpy())
    
    print("Plotting t-SNE...")
    plot_tsne(embedding, np.array(pids))
    save_path = osp.join(args.logs_dir, 'tsne_visualization.jpg')
    plt.savefig(save_path)
    print(f"t-SNE plot saved to {save_path}")


def _load_single_image(image_path, height, width, self_norm):
    if not osp.isfile(image_path):
        print(f"Error: Image path not found: {image_path}")
        return None
    
    if self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    transform = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def visualize_heatmap(model, args):
    print("==> Starting Heatmap visualization...")
    if not args.vis_image:
        print("Error: --vis_image path is required for heatmap visualization.")
        return

    img_tensor = _load_single_image(args.vis_image, args.height, args.width, args.self_norm)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(device)
    
    # Logic from TCMM/evaluators_heatmap.py
    m = model.module if isinstance(model, nn.DataParallel) else model
    patch_size = m.patch_embed.patch_size
    
    # Forward through the model to get attention
    with torch.no_grad():
        y = m.forward_features(img_tensor)
        attentions = m.get_last_selfattention(y)

    nh = attentions.shape[1]
    w_featmap = img_tensor.shape[-2] // patch_size
    h_featmap = img_tensor.shape[-1] // patch_size
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # Save the last head's attention map
    fname = osp.basename(args.vis_image).split('.')[0]
    save_path = osp.join(args.logs_dir, f'{fname}_heatmap.png')
    plt.imsave(fname=save_path, arr=attentions[-1], format='png')
    print(f"Heatmap saved to {save_path}")


def visualize_dino_attention(model, args):
    print("==> Starting DINO attention visualization...")
    if not args.vis_image:
        print("Error: --vis_image path is required for DINO attention visualization.")
        return

    img_tensor = _load_single_image(args.vis_image, args.height, args.width, args.self_norm)
    if img_tensor is None:
        return
        
    img_tensor = img_tensor.to(device)
    m = model.module if isinstance(model, nn.DataParallel) else model
    patch_size = m.patch_embed.patch_size

    # make the image divisible by the patch size
    w, h = img_tensor.shape[2] - img_tensor.shape[2] % patch_size, img_tensor.shape[3] - img_tensor.shape[3] % patch_size
    img_tensor = img_tensor[:, :, :w, :h]

    w_featmap = img_tensor.shape[-2] // patch_size
    h_featmap = img_tensor.shape[-1] // patch_size

    with torch.no_grad():
        attentions = m.get_last_selfattention(img_tensor.to(device))

    nh = attentions.shape[1] # number of heads

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    fname_base = osp.basename(args.vis_image).split('.')[0]
    save_dir = osp.join(args.logs_dir, f'{fname_base}_dino_attention')
    os.makedirs(save_dir, exist_ok=True)
    
    torchvision.utils.save_image(torchvision.utils.make_grid(img_tensor, normalize=True, scale_each=True), osp.join(save_dir, "img.png"))
    for j in range(nh):
        fname = osp.join(save_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
    
    print(f"DINO attention maps saved to {save_dir}")

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Visualization Functions End
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, nv_root=dataset.images_dir, transform=train_transformer), 
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=False), length=iters)  # shuffle=not rmgs_flag
    return train_loader


def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)
    else:
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem, feat_fusion=args.feat_fusion, multi_neck=args.multi_neck)
    # use CUDA
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(args, dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # If we are just visualizing, no need to train, just load model and run visualization
    if args.visualize:
        print('==> Loading best model for visualization')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        
        if args.visualize == 'tsne':
            visualize_tsne(model, test_loader, args)
        elif args.visualize == 'heatmap':
            if 'heatmap' not in args.arch:
                 print("Warning: Heatmap visualization is best with 'vit_small_heatmap' architecture.")
            visualize_heatmap(model, args)
        elif args.visualize == 'dino_attention':
            visualize_dino_attention(model, args)
        
        end_time = time.monotonic()
        print('Total running time for visualization: ', timedelta(seconds=end_time - start_time))
        return # Exit after visualization

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    print('optimizer: %s'%(args.optimizer))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer(model)
    print('==> Initialize features memory')
    m = model.module if isinstance(model, nn.DataParallel) else model
    feature_memory = ClusterMemory(m.num_features, len(dataset.train), temp=args.temp,
                                   momentum=args.momentum, use_hard=args.use_hard).to(device)
    cluster_loader = get_test_loader(args, dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))

    # create index corresponding dic
    index_dic = {}
    for it, data in enumerate(cluster_loader):
        _, path_list, _, _, indexes = data
        for i in range(len(path_list)):
            file_path = path_list[i]
            file_name = osp.basename(file_path)
            index_dic[file_name] = indexes[i]

    features, _ = extract_features(model, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
    feature_memory.features = F.normalize(features, dim=1).to(device)
    trainer.feature_memory = feature_memory
    # DBSCAN cluster init
    eps = args.eps
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
    for epoch in range(args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            # select & cluster images as training set of this epochs
            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
            pseudo_labels = cluster.fit_predict(rerank_dist)
            feature_memory.labels = torch.Tensor(pseudo_labels).to(device)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        del features

        # Create ClusterMemory
        memory = ClusterMemory(m.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).to(device)
        memory.features = F.normalize(cluster_features, dim=1).to(device)

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer, args.K, args.patch_rate, args.positive_rate, index_dic=index_dic,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='VeRi',
                        choices=datasets.names())
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-K', type=int, default=64, help="negative samples number for instance memory")
    parser.add_argument('--patch-rate', type=float, default=0.4, help="noise patch rate for patch refine")
    parser.add_argument('--positive-rate', type=int, default=3, help="positive sample number for patch refine")
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each i dentity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # DBSCAN
    parser.add_argument('--eps', type=float, default=0.7,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='vit_small',
                        choices=models.names())
    parser.add_argument('-pp', '--pretrained-path', type=str, default='../pass_imagenet_lup.pth')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the memory")
    #vit
    parser.add_argument('--drop-path-rate', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', default=True)
    parser.add_argument('--conv-stem', action="store_true")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default='../logs')
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--feat-fusion', type=str, default='cat')
    # parser.add_argument('--multi-neck', default=True)
    parser.add_argument('--multi-neck', action="store_true")
    parser.add_argument('--use-hard', default=True)
    
    # Visualization arguments
    parser.add_argument('--visualize', type=str, choices=['tsne', 'heatmap', 'dino_attention'], default=None,
                        help='Select visualization to generate after training/loading model.')
    parser.add_argument('--vis_image', type=str, default=None,
                        help='Path to a single image for heatmap or DINO attention visualization.')

    main()

# =================================================================================
# 腳本使用說明 (How to Use This Script)
# =================================================================================
#
# 1. 訓練模型 (To Train the model):
#    直接執行此腳本，不要加上 --visualize 參數。
#    `python train/main.py`
#    程式將會進行訓練，並將最佳模型儲存於日誌 (logs) 目錄下。
#
# 2. 生成視覺化圖表 (To Generate Visualizations):
#    在模型訓練完成後 (日誌目錄中已有 'model_best.pth.tar' 檔案)，使用 --visualize 參數。
#    程式會跳過訓練，直接載入最佳模型來生成指定的圖表。
#
#    A. t-SNE 分佈圖:
#       此功能會分析整個測試集的特徵分佈。
#       `python train/main.py --visualize tsne`
#       輸出: `tsne_visualization.jpg` 將會儲存在日誌目錄下。
#
#    B. Heatmap 熱力圖:
#       為單張圖片生成注意力熱力圖，需要搭配 --vis_image 參數指定圖片路徑。
#       `python train/main.py --visualize heatmap --vis_image "path/to/your/image.jpg"`
#       輸出: `<image_name>_heatmap.png` 將會儲存在日誌目錄下。
#
#    C. DINO 自注意力圖:
#       為單張圖片生成 ViT 模型每個 head 的自注意力圖，需要搭配 --vis_image 參數。
#       `python train/main.py --visualize dino_attention --vis_image "path/to/your/image.jpg"`
#       輸出: 日誌目錄下會建立一個新的資料夾，內含所有注意力圖。
#
# =================================================================================