
## Matrix Light ReID

This is a Person-ReID training framework based on [mmcls](https://github.com/open-mmlab/mmclassification), which supports both ResNet family backbones and mobile level backbones, with modular design.

We also support network searching to find best accuracy-latency tradeoff, for lightweight models.


## Performance

| Backbone | Head        | mAP   | Rank-1 |
|----------|-------------|-------|--------|
| ResNet50 | BagOfTricks | 85.53 | 93.91  |
|          |             |       |        |
|          |             |       |        |

[more]

## Data Prepare

Unzip Market1501 dataset to ``data/Market-1501-v15.09.15/``


## Training


For training bag-of-tricks model:

``CUDA_VISIBLE_DEVICES='0' PORT=29711 bash tools/dist_train.sh ./configs/reid/resnet50_market1501.py 1``


## Evaluation

As default, the training process will evaluate the scores of mAP and Rank-1 every 10 epochs.


## Network Search for Lightweight backbones

[todo]

## Reference

Some code is inspired by [fastreid](https://github.com/JDAI-CV/fast-reid) and [gluon-reid](https://github.com/xiaolai-sqlai/gluon-reid)


Ref-Papers:

```
@inproceedings{luo2019bag,
  title={Bag of tricks and a strong baseline for deep person re-identification},
  author={Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={0--0},
  year={2019}
}
```
