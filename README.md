
## Matrix Light ReID (MLR)

MLR is a Person-ReID training framework based on [mmcls](https://github.com/open-mmlab/mmclassification), which supports both ResNet family backbones and mobile level backbones, with modular design.

We also support network searching to find best accuracy-latency tradeoff, for lightweight models.


## Baseline Performance

| Backbone | Head        | mAP   | Rank-1 | FLOPs |
|----------|-------------|-------|--------|-------|
| ResNet50 | BagOfTricks | 85.53 | 93.91  | 4.08G |
|          |             |       |        |       |


[more]

## Data Prepare

Unzip Market1501 dataset to ``data/Market-1501-v15.09.15/``


## Training


For training bag-of-tricks model:

``CUDA_VISIBLE_DEVICES='0' PORT=29711 bash tools/dist_train.sh ./configs/reid/resnet50_market1501.py 1``


## Evaluation

As default, the training process will evaluate the scores of mAP and Rank-1 every 10 epochs.


## Network Search for Lightweight backbones

As for the name **Light** ReID, we try to find a lightweight solution to produce high quality ReID features.

Here we use the MobileNetV2 as backbone and the same optimizer from ResNet50 baseline as our base solution. Our aim is to find a better backbone and training strategy to obtain good ReID mAP and Rank-1 accuracy, under the train-from-scratch situation.

| Backbone    | Optimizer               | LR epoch steps  | mAP   | Rank-1 | FLOPs |
| ----------- | ----------------------- | --------------- | ----- | ------ | ----- |
| MobileNetV2 | Adam, wd 5e-4           | 50,120,150      | 46.96 | 71.17  | 210M  |
| MobileNetV2 | SGD, wd 1e-4            | 50,120,150      | 47.70 | 71.61  | 210M  |
| MobileNetV2 | SGD, wd 5e-4            | 50,120,150      | 61.62 | 81.32  | 210M  |
| MLR-210M-A  | SGD, wd 5e-4            | 50,120,150      | 64.57 | 84.56  | 212M  |
| MLR-210M-B  | SGD, wd 5e-4            | 50,120,150      | 65.23 | 83.88  | 207M  |
| MLR-210M-A* | SGD, wd 5e-4, Warmup-2K | 150,250,320,350 |       |        | 212M  |
| MLR-210M-B* | SGD, wd 5e-4, Warmup-2K | 150,250,320,350 |       |        | 207M  |

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
