import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.losses import Accuracy
from ..builder import HEADS, build_loss
from .base_head import BaseHead


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
            self.bnneck = nn.BatchNorm1d(in_channels)
            nn.init.constant_(self.bnneck.weight, 1)
            self.bnneck.weight.requires_grad = False
            neck.append(self.bnneck)

        self.neck = nn.Sequential(*neck)
        #self.neck.apply(weights_init_kaiming)

        self.loss_ce = build_loss(loss_ce)
        self.loss_tri = None
        if loss_tri is not None:
            self.loss_tri = build_loss(loss_tri)

        #self.compute_loss = build_loss(loss)
        #self.compute_accuracy = Accuracy(topk=self.topk)

    def extract_feats(self, x):
        xbn = self.neck(x)
        return (x, xbn)

    def loss(self, feats , gt_label):
        assert len(feats)==2 #gap layer and bn layer
        feat = feats[0]
        feat_bn = feats[1]
        num_samples = len(feat)
        losses = dict()
        # compute loss
        losses['loss_ce'] = self.loss_ce(feat_bn, gt_label, avg_factor=num_samples)
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
