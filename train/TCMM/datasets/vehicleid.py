from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..utils.data import BaseImageDataset
import os

class VehicleID(BaseImageDataset):
    """
    VehicleID
    Reference:
    Liu, H., Tian, Y., Wang, Y., Pang, L., & Huang, T. (2016). Deep relative distance learning: Tell the difference between similar vehicles. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2167-2175).
    URL: https://pkuml.org/resources/pku-vehicleid.html

    Dataset statistics:
    # identities: 13164 (train) + 800 (test)
    # images: 113346 (train) + 17638 (test)
    """
    dataset_dir = 'VehicleID'

    def __init__(self, root, verbose=True, test_size=800, **kwargs):
        super(VehicleID, self).__init__()
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split', 'train_list.txt')
        self.test_list_file = osp.join(self.dataset_dir, 'train_test_split', 'test_list_{}.txt'.format(test_size))
        
        self._check_before_run()
        
        train = self._process_split(self.train_list, relabel=True)
        
        # Split test set into query and gallery
        test_data = self._process_split(self.test_list_file, relabel=False)
        query = []
        gallery = []
        pid_dict = {}
        for img_path, pid, camid in test_data:
            if pid not in pid_dict:
                query.append((img_path, pid, camid))
                pid_dict[pid] = True
            else:
                gallery.append((img_path, pid, camid))

        if verbose:
            print("=> VehicleID loaded (test size: {})".format(test_size))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.isdir(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.isdir(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not osp.isfile(self.train_list):
            raise RuntimeError("'{}' is not available".format(self.train_list))
        if not osp.isfile(self.test_list_file):
            raise RuntimeError("'{}' is not available".format(self.test_list_file))

    def _process_split(self, list_file, relabel=False):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        dataset = []
        pid_container = set()
        for line in lines:
            img_name, pid = line.strip().split()
            pid = int(pid)
            pid_container.add(pid)
            img_path = osp.join(self.img_dir, img_name + '.jpg')
            # Camera ID is not available in VehicleID, so we use a dummy value
            camid = -1 
            dataset.append((img_path, pid, camid))
        
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
            for i in range(len(dataset)):
                img_path, pid, camid = dataset[i]
                dataset[i] = (img_path, pid2label[pid], camid)

        return dataset
