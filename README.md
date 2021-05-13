
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
@misc{luo2019bag,
      title={Bag of Tricks and A Strong Baseline for Deep Person Re-identification}, 
      author={Hao Luo and Youzhi Gu and Xingyu Liao and Shenqi Lai and Wei Jiang},
      year={2019},
      eprint={1903.07071},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
