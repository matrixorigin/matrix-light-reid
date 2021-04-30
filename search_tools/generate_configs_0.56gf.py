
import os
import os.path as osp
import io
import numpy as np
import argparse
import datetime
import importlib
import configparser
from tqdm import tqdm

import torch
import autotorch as at
from mmcv import Config

from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


@at.obj(
    layer0_channels=at.Int(8,64),
    layer1_channels=at.Int(8,64),
    last_stride=at.Int(1,2),
    stage_blocks=at.List(
        at.Int(1,6),
        at.Int(1,6),
        at.Int(1,6),
        at.Int(1,6),
        ),
    stage_expands=at.List(
        at.Int(2,6),
        at.Int(2,6),
        at.Int(2,6),
        at.Int(2,6),
        ),
    stage_planes_ratio=at.List(
        at.Real(1.0,4.0),
        at.Real(1.0,4.0),
        at.Real(1.0,4.0),
        at.Real(1.0,4.0),
        at.Real(1.0,4.0),
        ),
)
class GenConfig1:
    def __init__(self, **kwargs):
        d = {}
        d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        self.m = 1.0

    def stage_blocks_multi(self, m):
        self.m = m

    def merge_cfg(self, base_cfg):

        self.layer0_channels = max(8, int(self.layer0_channels*self.m)//8 * 8)
        self.layer1_channels = max(8, int(self.layer1_channels*self.m)//8 * 8)
        stage_planes = [self.layer1_channels]
        arch_settings = [ 
                [1, self.layer0_channels, 1, 2],
                [1, self.layer1_channels, 1, 1],
                ]
        for i in range(4):
            stride = 2 if i<3 else self.last_stride
            ratio = self.stage_planes_ratio[i]
            planes = int(arch_settings[-1][1] * ratio) //8 * 8
            setting = [self.stage_expands[i], planes, self.stage_blocks[i], stride]
            arch_settings.append(setting)

        planes = int(arch_settings[-1][1] * self.stage_planes_ratio[-1]) //8 * 8
        setting = [1, planes, 1, 1]
        arch_settings.append(setting)
        base_cfg['model']['backbone']['arch_settings'] = arch_settings
        base_cfg['model']['head']['in_channels'] = planes
        feat_dim = planes
        return base_cfg, feat_dim

def get_args():
    # data settings
    parser = argparse.ArgumentParser(description='Auto-LightReID')
    # config files
    parser.add_argument('--group', type=str, default='configs/lightreid_0.56gf', help='configs work dir')
    parser.add_argument('--template', type=int, default=0, help='template config index')
    parser.add_argument('--write-from', type=int, default=-1, help='write index from')
    parser.add_argument('--gflops', type=float, default=0.56, help='expected flops')
    #parser.add_argument('--mode', type=int, default=1, help='generation mode, 1 for searching backbone, 2 for search all')
    # input size
    # target flops
    parser.add_argument('--eps', type=float, default=2e-2,
                         help='eps for expected flops')
    # num configs
    parser.add_argument('--num-configs', type=int, default=32, help='num of expected configs')
    parser = parser

    args = parser.parse_args()
    return args

def get_flops(cfg, input_shape):
    model = build_classifier(cfg.model)
    model.eval()
    if hasattr(model, 'extract_feat'):
        model.forward = model.extract_feat
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))
    buf = io.StringIO()
    all_flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=True, as_strings=False, ost=buf)
    buf = buf.getvalue()

    return all_flops/1e9
    #print('FLOPs:', flops/1e9)
    #return flops <= (1. + eps) * target_flops and \
    #    flops >= (1. - eps) * target_flops

def is_flops_valid(flops, target_flops, eps):
    return flops <= (1. + eps) * target_flops and \
        flops >= (1. - eps) * target_flops

def main():
    args = get_args()
    print(datetime.datetime.now())

    input_shape = (3,256,128)
    runtime_input_shape = input_shape

    assert osp.exists(args.group)
    group_name = args.group.split('/')[-1]
    assert len(group_name)>0
    input_template = osp.join(args.group, "%s_%d.py"%(group_name, args.template))
    assert osp.exists(input_template)
    assert args.write_from!=0
    if args.write_from<=0:
        write_index = args.template+1
        while True:
            output_cfg = osp.join(args.group, "%s_%d.py"%(group_name, write_index))
            if not osp.exists(output_cfg):
                break
            write_index+=1
    else:
        write_index = args.write_from
    print('write-index from:', write_index)

    gen = GenConfig1()



    pp = 0
    write_count = 0
    while write_count < args.num_configs:
        pp+=1
        base_cfg = Config.fromfile(input_template)
        config = gen.rand
        cls_cfg, feat_dim = config.merge_cfg(base_cfg)
        all_flops = get_flops(cls_cfg, runtime_input_shape)
        is_valid = True
        if pp%10==0:
            print(pp, all_flops, datetime.datetime.now())
        if feat_dim<1024 or feat_dim>2048:
            continue
        #if args.mode==1:
        #    if np.argmax(backbone_flops)!=1:
        #        continue
        #    if np.mean(backbone_flops[1:3])*0.8<np.mean(backbone_flops[-2:]):
        #        continue
        if not is_flops_valid(all_flops, args.gflops, args.eps):
            continue

        output_cfg_file = osp.join(args.group, "%s_%d.py"%(group_name, write_index))
        cls_cfg.dump(output_cfg_file)
        print('SUCC', write_index, all_flops, datetime.datetime.now())
        write_index += 1
        write_count += 1

if __name__ == '__main__':
    main()

