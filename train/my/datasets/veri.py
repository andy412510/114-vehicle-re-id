# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from ..utils.data import BaseImageDataset


class VeRi(BaseImageDataset):
    """VeRi.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "VeRi"
    dataset_name = "veri"

    def __init__(self, root='datasets', **kwargs):
        super(VeRi, self).__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir, is_train=False)
        gallery = self._process_dir(self.gallery_dir, is_train=False)
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data