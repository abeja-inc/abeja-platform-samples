SegNet
======

This is a SegNet [1,2] model example for semantic segmenation. This code is based on [chainercv](https://github.com/chainer/chainercv) example.

## Train

### Dataset preperation

The CamVid dataset example.

https://github.com/abeja-inc/abeja-platform-samples/tree/master/dataset/camvid

### Training

The training setting sample is here. `{JOB_NAME}` is arbitrarily name that you can define. `{CamVid-train_ID}`, `{CamVid-val_ID}` are described in ABEJA Platform Console.
```
name: {JOB_NAME}
handler: train:handler
image: abeja-inc/deepgpu:0.1.0
datasets:
  train: '{CamVid-train_ID}'
  val: '{CamVid-val_ID}'
params:
  USE_GPU: '0'
```

You can train the model with the following code using ABEJA CLI. 
```
$ abeja training create-job-definition
$ abeja training create-version
$ abeja training create-job --version {JOB_VERSION}
```

`{JOB_VERSION}` is target code version. You can get it from output of `create-version` or ABEJA Platform Console.

## Evaluation

Not yet.

## Prediction

Coming soon.

## NOTE

- According to the original implementation, the authors performed LR flipping to the input images for data augmentation: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/dense_image_data_layer.cpp#L168-L175
- Chainer's LRN layer is different from Caffe's one in terms of the meaning of "alpha" argment, so we modified the Chainer's LRN default argment to make it same as Caffe's one: https://github.com/alexgkendall/caffe-segnet/blob/segnet-cleaned/src/caffe/layers/lrn_layer.cpp#L121

## Experimental settings

We used the completely same parameters for all settings.

| Implementation | Optimizer   | Learning rage | Momentum | Weight decay | Model code |
|:--------------:|:-----------:|:-------------:|:--------:|:------------:|:----------:|
| ChainerCV      | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic.py](https://github.com/chainer/chainercv/tree/master/chainercv/links/model/segnet/segnet_basic.py) |
| Official       | MomentumSGD | 0.1           | 0.9      | 0.0005       | [segnet_basic_train.prototxt](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_basic_train.prototxt) |

## Comparizon with the paper results

| Implementation | Global accuracy | Class accuracy | mean IoU   |
|:--------------:|:---------------:|:--------------:|:----------:|
| ChainerCV      | 82.7 %          | **67.1 %**     | **49.4 %** |
| Official       | **82.8 %**      | 62.3%          | 46.3 %     |

The evaluation can be conducted using [`chainercv/examples/semantic_segmentation/eval_cityscapes.py`](https://github.com/chainer/chainercv/blob/master/examples/semantic_segmentation).


# Reference

1. Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." PAMI, 2017.
2. Vijay Badrinarayanan, Ankur Handa and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling." arXiv preprint arXiv:1505.07293, 2015.
3. [Original implementation 1](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)
4. [Original implementation 2](https://github.com/chainer/chainercv/tree/master/examples/segnet)
