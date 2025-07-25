from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch


def anchor(batch_input, batch_labels, indexes, feature_memory, k, temp, momentum):
    """
    The anchor contrastive loss implementation in our paper.(Andy Zhu)
    The idea is to find the hardest same cluster sample as positive sample. (the minimum cosine similarity)
    And find K hard different cluster samples as negative samples. (the maximum cosine similarity)
    Finally, update feature memory by momentum update.
    """
    instance_m = feature_memory.features.clone().detach()
    mat = torch.matmul(batch_input, instance_m.transpose(0, 1))
    positives = []
    negatives = []
    for i in range(batch_labels.size(0)):
        pos_labels = (feature_memory.labels == batch_labels[i])
        pos = mat[i, pos_labels]
        positives.append(pos[torch.randint(0, pos.size(0), (1,))])
        neg_labels = (feature_memory.labels != batch_labels[i])  # pseudo labels w/ -1
        # neg_labels = torch.logical_and(feature_memory.labels != batch_labels[i], feature_memory.labels != -1)  # ignore -1
        indices = torch.randperm(mat[i, neg_labels].size(0))
        neg = mat[i, neg_labels][indices]
        # neg = torch.sort(mat[i, neg_labels], descending=True)[0]
        idx = neg[:k]
        negatives.append(idx)
    positives = torch.stack(positives)
    positives = positives.view(-1,1)
    negatives = torch.stack(negatives)
    anchor_out = torch.cat((positives, negatives), dim=1) / temp

    with torch.no_grad():
        for data, index in zip(batch_input, indexes):
            feature_memory.features[index] = momentum * feature_memory.features[index] + (1.-momentum) * data
            feature_memory.features[index] /= feature_memory.features[index].norm()
    return anchor_out


class Trainer(object):
    def __init__(self, encoder, memory=None, feature_memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.feature_memory = feature_memory
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch, data_loader, optimizer, k, temp, momentum, index_dic, print_freq=10, train_iters=400, ):
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
            inputs, targets, indexes, path_list = self._parse_data(inputs)
            contrast_targets = torch.zeros([targets.size(0)]).cuda().long()
            # correct index
            for j in range(len(path_list)):
                file_path = path_list[j]
                file_name = file_path.split('/')[-1]
                indexes[j] = index_dic[file_name]
            # forward
            f_out = self._forward(inputs)
            output = anchor(f_out[0], targets, indexes, self.feature_memory, k, temp, momentum)
            loss = self.criterion(output, contrast_targets)

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

    def UnNormalize(self, tensor):
        invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                       transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                            std=[1., 1., 1.]),
                                       ])

        inv_tensor = invTrans(tensor)
        return inv_tensor