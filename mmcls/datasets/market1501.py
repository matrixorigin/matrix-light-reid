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

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def get_samples(root, extensions, test_mode):
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
                cid = int(fn.split('_')[1][1])
                sample = (fn, pid, cid)
                samples.append(sample)
    return samples

def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

@DATASETS.register_module()
class Market1501(BaseDataset):
    IMG_EXTENSIONS = ('.jpg',)
    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None,
                 ann_file=None,
                 test_mode=False,
                 triplet_sampler=False):
        super(Market1501, self).__init__(data_prefix, pipeline, classes, ann_file, test_mode)
        self.triplet_sampler = triplet_sampler
        #print('TTTTTT:', self.triplet_sampler)

    def load_annotations(self):
        if not self.test_mode:
            dir_prefix = osp.join(self.data_prefix, 'bounding_box_train')
            samples = get_samples(
                dir_prefix,
                extensions=self.IMG_EXTENSIONS, test_mode=self.test_mode)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{dir_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.samples = samples

            data_infos = []
            min_pid = 1e10
            max_pid = 0
            for filename, pid, cid in self.samples:
                #print(filename, gt_label)
                info = {'img_prefix': dir_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(pid, dtype=np.int64)
                data_infos.append(info)
                min_pid = min(min_pid, pid)
                max_pid = max(max_pid, pid)
            #print('PID-STAT:', min_pid, max_pid)
            return data_infos
        else:
            dir_prefix = osp.join(self.data_prefix, 'bounding_box_test')
            samples_test = get_samples(
                dir_prefix,
                extensions=self.IMG_EXTENSIONS, test_mode=self.test_mode)
            if len(samples_test) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{dir_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))
            self.test_size = len(samples_test)
            test_prefix = dir_prefix
            dir_prefix = osp.join(self.data_prefix, 'query')
            samples_query = get_samples(
                dir_prefix,
                extensions=self.IMG_EXTENSIONS, test_mode=self.test_mode)
            if len(samples_query) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{dir_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))
            self.query_size = len(samples_query)
            query_prefix = dir_prefix
            #print('TTT:', self.test_size, self.query_size)

            self.samples = samples_test + samples_query
            data_infos = []
            for idx, (filename, pid, cid) in enumerate(self.samples):
                #print(filename, gt_label)
                if idx<self.test_size:
                    info = {'img_prefix': test_prefix}
                else:
                    info = {'img_prefix': query_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(pid, dtype=np.int64)
                info['cid'] = np.array(cid, dtype=np.int64)
                data_infos.append(info)
            return data_infos

    def __evaluate(self,
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
            'map', 'rank1'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        cams = np.array([data['cid'] for data in self.data_infos])
        num_imgs = len(results)
        #print('XXXX', results.shape, gt_labels.shape, num_imgs)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'
        assert (self.test_size+self.query_size) == num_imgs, 'dataset testing results should '\
            'be of the same as data size.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')
        features = results
        features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))

        test_feature = features[:self.test_size]
        query_feature = features[self.test_size:]
        test_label = gt_labels[:self.test_size]
        query_label = gt_labels[self.test_size:]
        test_cam = cams[:self.test_size]
        query_cam = cams[self.test_size:]
        num = query_label.size
        dist_all = np.dot(query_feature, test_feature.T)

        CMC = np.zeros(test_label.size)
        ap = 0.0
        for i in range(num):
            cam = query_cam[i]
            label = query_label[i]
            index = np.argsort(-dist_all[i])

            query_index = np.argwhere(test_label==label)
            camera_index = np.argwhere(test_cam==cam)

            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            junk_index = np.intersect1d(query_index, camera_index)
        
            ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
            CMC = CMC + CMC_tmp
            ap += ap_tmp

        CMC = CMC/num #average CMC
        rank1 = CMC[0]
        mAP = ap/num
        if 'map' in metrics:
            eval_results['map'] = mAP
        if 'rank1' in metrics:
            eval_results['rank1'] = rank1
        #print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))
        return eval_results

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
            'map', 'rank1'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        cams = np.array([data['cid'] for data in self.data_infos])
        num_imgs = len(results)
        #print('XXXX', results.shape, gt_labels.shape, num_imgs)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'
        assert (self.test_size+self.query_size) == num_imgs, 'dataset testing results should '\
            'be of the same as data size.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')
        features = results
        features = features / np.sqrt(np.sum(np.square(features), axis=1, keepdims=True))

        gallery_features = features[:self.test_size]
        query_features = features[self.test_size:]
        gallery_pids = gt_labels[:self.test_size]
        query_pids = gt_labels[self.test_size:]
        gallery_camids = cams[:self.test_size]
        query_camids = cams[self.test_size:]


        #dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)
        dist = 1.0 - np.dot(query_features, gallery_features.T)


        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        #for r in [1, 5, 10]:
        #    self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100



        if 'map' in metrics:
            eval_results['map'] = mAP * 100
        if 'rank1' in metrics:
            eval_results['rank1'] = cmc[0]*100
        #print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))
        return eval_results
