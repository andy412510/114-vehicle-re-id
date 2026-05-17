# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Some parts are taken from https://github.com/Liusifei/UVC
"""
import os
import copy
import glob
import queue
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import utils


@torch.no_grad()
def eval_video_tracking_davis(args, model, frame_list, video_dir, first_seg, seg_ori, color_palette):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    video_folder = os.path.join(args.output_dir, os.path.basename(video_dir))
    os.makedirs(video_folder, exist_ok=True)

    # The queue stores the n preceeding frames
    que = queue.Queue(args.n_last_frames)

    # first frame
    frame1, ori_h, ori_w = utils.read_frame(frame_list[0], args.patch_size)
    frame1 = frame1.cuda(non_blocking=True)

    # extract first frame feature
    # out = model(frame1)
    out = model.get_intermediate_layers(frame1, n=1)[0]
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame1.shape[2] / args.patch_size), int(frame1.shape[3] / args.patch_size)
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    f1 = out.permute(2, 0, 1)

    que.put(f1)
    
    # saving first segmentation
    imwrite_indexed(os.path.join(video_folder, "00000.png"), seg_ori, color_palette)
    
    # n_last_frames features
    last_feats = [f1]

    # n_last_frames gt segmentations
    last_segs = [first_seg]

    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar, _, _ = utils.read_frame(frame_list[cnt], args.patch_size)
        frame_tar = frame_tar.cuda(non_blocking=True)

        # out = model(frame_tar)
        out = model.get_intermediate_layers(frame_tar, n=1)[0]
        out = out[:, 1:, :]  # we discard the [CLS] token
        f_tar = out[0].reshape(h, w, dim)
        f_tar = f_tar.permute(2, 0, 1)

        # compute attention (corresponds to the ``labels propagation'' in our paper)
        # we compute it for each of the n_last_frames and then we average them
        frame_tar_avg = 0
        for i in range(len(last_feats)):
            # compute affinity between the target frame and the i-th preceding frame
            # (h*w, h*w)
            affinity = torch.mm(f_tar.reshape(dim, h * w).t(), last_feats[i].reshape(dim, h * w))
            affinity = torch.nn.functional.softmax(affinity / args.temperature, dim=1)
            
            # propagate the segmentation (h*w, K)
            frame_tar_avg += torch.mm(affinity, last_segs[i].reshape(-1, h * w).t())

        frame_tar_avg /= len(last_feats)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = os.path.basename(frame_list[cnt]).replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


def restrict_neighborhood(h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    # This is done by zeroing out the affinity matrix for nodes that are too far away
    # In the DAVIS paper, this is done by considering only nodes in a 15x15 window
    # In our paper, we consider a larger window of 40x40
    # (h*w, h*w)
    mask = torch.zeros(h * w, h * w).cuda()
    for i in range(h):
        for j in range(w):
            for i_ in range(max(0, i - 15), min(h, i + 16)):
                for j_ in range(max(0, j - 15), min(w, j + 16)):
                    mask[i * w + j, i_ * w + j_] = 1
    return mask


def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png from array with color palette """
    if np.atleast_3d(array).shape[2] != 1:
      raise ValueError("Only 2D arrays can be saved as indexed png")

    # The array must be uint8
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    img = Image.fromarray(array, mode='P')
    img.putpalette(color_palette)
    img.save(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with video segmentation propagation')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature for the affinity matrix')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--n_last_frames', default=7, type=int, help='Number of preceding frames to consider.')
    parser.add_argument('--dataset_path', default='/path/to/davis/', type=int, help='Path to DAVIS dataset.')
    parser.add_argument('--output_dir', default='.', help='Path where to save segmentations')
    parser.add_argument('--model_path', default='/path/to/model.pth', help='Path to the model to evaluate.')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT architectures).')
    args = parser.parse_args()

    # ============ building network ... ============
    # if the network is a Vision Transformer
    if args.arch in models.__dict__.keys():
        model = models.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    else:
        print(f"Unknow architecture: {args.arch}")

    model.cuda()
    model.eval()

    # load weights to evaluate
    utils.load_pretrained_weights(model, args.model_path, "teacher", args.arch, args.patch_size)

    # ============ evaluation ... ============
    color_palette = []
    for i in range(256):
        color_palette.extend((i, i, i))
    color_palette[:3*len(utils.DAVIS_COLOR_PALETTE)] = utils.DAVIS_COLOR_PALETTE

    # list videos
    video_list = glob.glob(os.path.join(args.dataset_path, "JPEGImages", "480p", "*"))
    for video_dir in video_list:
        frame_list = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
        first_seg_path = os.path.join(args.dataset_path, "Annotations", "480p", os.path.basename(video_dir), "00000.png")
        first_seg = np.array(Image.open(first_seg_path))
        ori_h, ori_w = first_seg.shape
        first_seg = np.array(Image.fromarray(first_seg).resize((ori_w // args.patch_size, ori_h // args.patch_size), 0))
        first_seg = torch.from_numpy(first_seg).cuda()
        first_seg = torch.nn.functional.one_hot(first_seg.long(), num_classes=256).permute(2, 0, 1).float()
        
        eval_video_tracking_davis(args, model, frame_list, video_dir, first_seg, first_seg, color_palette)
