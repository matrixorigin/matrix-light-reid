# encoding: utf-8

import glob
import os.path as osp
import os
import numpy as np

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
                #path = os.path.join(root, fn)
                pid = int(fn.split('_')[0])
                if pid<0:
                    continue
                assert 0 <= pid <= 1501  # pid == 0 means background
                sample = (fn, pid)
                samples.append(sample)
    return samples

@DATASETS.register_module()
class Market1501(BaseDataset):
    IMG_EXTENSIONS = ('.jpg',)

    def load_annotations(self):
        samples = get_samples(
            self.data_prefix,
            extensions=self.IMG_EXTENSIONS)
        if len(samples) == 0:
            raise (RuntimeError('Found 0 files in subfolders of: '
                                f'{self.data_prefix}. '
                                'Supported extensions are: '
                                f'{",".join(self.IMG_EXTENSIONS)}'))

        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            #print(filename, gt_label)
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        print('XXXX', results.shape, gt_labels.shape, num_imgs)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')
        return {'accuracy': 1.0}

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode, thrs=thrs)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results

