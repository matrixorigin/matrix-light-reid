# encoding: utf-8

import glob
import os.path as osp

from .base_dataset import BaseDataset
from .builder import DATASETS

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_samples(root, extensions):
    """Make dataset by walking all images under a root.

    Args:
        root (string): root directory of folders
        folder_to_idx (dict): the map from class name to class idx
        extensions (tuple): allowed extensions

    Returns:
        samples (list): a list of tuple where each element is (image, label)
    """
    samples = []
    root = os.path.expanduser(root)

    for _, _, fns in sorted(os.walk(root)):
        for fn in sorted(fns):
            if has_file_allowed_extension(fn, extensions):
                path = os.path.join(root, fn)
                pid = int(fn.split('_')[0])
                if pid<0:
                    continue
                assert 0 <= pid <= 1501  # pid == 0 means background
                sample = (path, pid)
                samples.append(sample)
    return samples

@DATASETS.register_module()
class Market1501(BaseDataset):
    IMG_EXTENSIONS = ('.jpg',)

    def load_annotations(self):
        samples = get_samples(
            self.data_prefix,
            folder_to_idx,
            extensions=self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: '
                                f'{self.data_prefix}. '
                                'Supported extensions are: '
                                f'{",".join(self.IMG_EXTENSIONS)}'))

        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

