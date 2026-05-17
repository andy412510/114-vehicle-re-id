from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
from ..utils.data import BaseImageDataset


class Blur(BaseImageDataset):
    """
    Blur dataset.
    """
    dataset_dir = 'blur/cropImgBlur'

    def __init__(self, root, verbose=True, **kwargs):
        super(Blur, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = self.dataset_dir
        self.query_dir = self.dataset_dir
        self.gallery_dir = self.dataset_dir

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Blur loaded")
            self.print_dataset_statistics(train, query, gallery)

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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.jpg'))
        pattern = re.compile(r'([a-zA-Z]+)(\d+)')

        pid_container = set()
        for img_path in img_paths:
            dir_name = osp.basename(osp.dirname(img_path))
            match = pattern.search(dir_name)
            if match:
                pid_container.add(int(match.group(2)))
        
        pid2label = {pid: label for label, pid in enumerate(sorted(list(pid_container)))}

        dataset = []
        for img_path in img_paths:
            file_name = osp.basename(img_path)
            dir_name = osp.basename(osp.dirname(img_path))
            match = pattern.search(dir_name)
            if not match:
                continue
            
            pid = int(match.group(2))
            
            if relabel:
                pid = pid2label[pid]
            
            # Try to find camera ID in filename
            # Patterns: cam1, c_3001, b_1001
            cam_match = re.search(r'(?:cam|c_|b_)(\d+)', file_name)
            if cam_match:
                camid_str = cam_match.group(1)
                if len(camid_str) >= 4:
                    camid = int(camid_str[0]) - 1
                else:
                    camid = int(camid_str) - 1
            else:
                camid = 0
            
            dataset.append((img_path, pid, camid))

        return dataset
