from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import torch

class Trainer(object):
    def __init__(self, encoder, memory=None, feature_memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.feature_memory = feature_memory

    def train(self, epoch, data_loader, optimizer, k, patch_rate, positive_rate, index_dic, print_freq=10, train_iters=400, vis_patch=False):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes, path_list = self._parse_data(inputs)
            # correct index
            for j in range(len(path_list)):
                file_path = path_list[j]
                file_name = file_path.split('/')[-1]
                indexes[j] = index_dic[file_name]
            # forward
            f_out = self._forward(inputs)
            loss, patch_feature = self.memory(f_out, labels, indexes, self.feature_memory, k, patch_rate, positive_rate)

            if vis_patch:
                self.vis_heatmap(inputs, path_list, patch_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, path_list, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), path_list

    def _forward(self, inputs):
        return self.encoder(inputs)

    def vis_heatmap(self, x, path_list, patch_features):
        batch = patch_features.size(0)
        invTrans = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                                        std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
        with torch.no_grad():
            for b in range(batch):
                file_name = os.path.basename(path_list[b])
                file_name_without_extension = os.path.splitext(file_name)[0]
                directory = f"./log/vis/patch/{file_name_without_extension}"
                os.makedirs(directory, exist_ok=True)
                fig, axes = plt.subplots(1, 11, figsize=(20, 5))

                img = x[b].unsqueeze(0)  # 提取個別影像並保持批次維度 (1, 3, H, W)
                img = invTrans(img)

                # 保存原始影像
                img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                axes[0].imshow(img)
                axes[0].axis('off')
                axes[0].set_title('Original')
                for i in range(10):
                    patch_feature = patch_features[b, i, :]
                    patch_heatmap = self.generate_heatmap(patch_feature[:128], grid_size=(16, 8))  # 取前128个特征
                    axes[i + 1].imshow(img)
                    axes[i + 1].imshow(patch_heatmap, cmap='jet', alpha=0.5)  # 使用 jet 颜色映射叠加热力图
                    axes[i + 1].axis('off')
                    axes[i + 1].set_title(f'Patch {i + 1}')
                plt.tight_layout()
                plt.savefig(os.path.join(directory, 'combined_heatmap.png'))
                plt.close()


    def generate_heatmap(self, patch_feature, grid_size=(8, 16)):
        heatmap = patch_feature.view(grid_size[0], grid_size[1]).cpu().detach().numpy()
        return heatmap

    def vis_patch_selec(self, x, path_list, positive_index, negative_index):
        with torch.no_grad():
            invTrans = transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                                            std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
            B = x.size(0)
            for b in range(B):
                file_name = os.path.basename(path_list[b])
                file_name_without_extension = os.path.splitext(file_name)[0]
                directory = f"./log/vis/patch/{file_name_without_extension}"
                os.makedirs(directory, exist_ok=True)

                img = x[b].unsqueeze(0)  # 提取個別影像並保持批次維度 (1, 3, H, W)
                img = invTrans(img)
                # 在應用卷積之前提取 patches
                patches = img.unfold(2, 16, 16).unfold(3, 16, 16)  # (1, 3, H/16, W/16, 16, 16)
                patches = patches.contiguous().view(1, 3, -1, 16, 16)  # (1, 3, (H/16)*(W/16), 16, 16)
                patches = patches.permute(0, 2, 1, 3, 4)  # (1, (H/16)*(W/16), 3, 16, 16)
                patches = patches.squeeze(0)  # ((H/16)*(W/16), 3, 16, 16)
                positive_patch = patches[positive_index[b]].permute(1, 2, 0).cpu().numpy()  # to HWC 格式

                img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(os.path.join(directory, 'original_image.png'))
                plt.close()
                plt.imshow(positive_patch)
                plt.axis('off')
                plt.savefig(os.path.join(directory, 'positive.png'))
                plt.close()

                negative_patch = patches[negative_index[b]]
                num_patches = negative_patch.shape[0]

                for i in range(num_patches):
                    patch = patches[i].permute(1, 2, 0).cpu().numpy()  # to HWC 格式
                    plt.imshow(patch)
                    plt.axis('off')
                    plt.savefig(os.path.join(directory, f'negative_{i}.png'))
                    plt.close()

            return patches