import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead
from mmcv.cnn import constant_init, kaiming_init


@HEADS.register_module()
class BotHead(BaseHead):
    """classification head.

    Args:
        loss (dict): Config of classification loss.
        topk (int | tuple): Top-k accuracy.
    """  # noqa: W605

    def __init__(self,
                 in_channels = 2048,
                 embedding_dim = 0,
                 num_classes = 1502,
                 loss_ce=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 loss_tri=dict(type='TripletLoss', loss_weight=1.0, margin=0.3, norm_feat=False),
                 bn_neck = True,
                 ):
        super(BotHead, self).__init__()

        neck = []
        feat_dim = in_channels
        if embedding_dim > 0:
            neck.append(nn.Conv2d(in_channels, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if bn_neck:
            self.bnneck = nn.BatchNorm1d(feat_dim)
            nn.init.constant_(self.bnneck.weight, 1)
            self.bnneck.weight.requires_grad = False
            neck.append(self.bnneck)

        self.neck = nn.Sequential(*neck)
        self.weight = nn.Parameter(torch.normal(0, 0.01, (num_classes, feat_dim)))


        self.loss_ce = build_loss(loss_ce)
        self.loss_tri = None
        if loss_tri is not None:
            self.loss_tri = build_loss(loss_tri)
        self._init_weights()

    def _init_weights(self):
        kaiming_init(self.neck)

    def extract_feats(self, x):
        xbn = self.neck(x)
        return (x, xbn)

    def loss(self, feats , gt_label):
        assert len(feats)==2 #gap layer and bn layer
        feat = feats[0]
        feat_bn = feats[1]
        num_samples = len(feat)
        logits = F.linear(feat_bn, self.weight)
        losses = dict()
        # compute loss
        losses['loss_ce'] = self.loss_ce(logits, gt_label, avg_factor=num_samples)
        if self.loss_tri is not None:
            losses['loss_tri'] = self.loss_tri(feat, gt_label, avg_factor=num_samples)
        # compute accuracy
        #acc = self.compute_accuracy(cls_score, gt_label)
        #assert len(acc) == len(self.topk)
        #losses['loss'] = loss
        #losses['accuracy'] = {f'top-{k}': a for k, a in zip(self.topk, acc)}
        return losses

    def forward_train(self, x, gt_label):
        feats = self.extract_feats(x)
        losses = self.loss(feats, gt_label)
        return losses

    def simple_test(self, x):
        feats = self.extract_feats(x)
        feat = feats[-1]
        feat = list(feat.detach().cpu().numpy())
        return feat
