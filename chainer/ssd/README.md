# Examples of Single Shot Multibox Detector [1]

## Train

### Dataset preperation

https://github.com/abeja-inc/abeja-platform-samples/tree/master/dataset/voc

### Training

The training setting sample is here. `{JOB_NAME}` is arbitrarily name that you can define. `{PascalVOC2007-trainval_ID}`, `{PascalVOC2012-trainval_ID}` and `{PascalVOC2007-test_ID}` are described in ABEJA Platform Console.
```
name: {JOB_NAME}
handler: train:handler
image: abeja-inc/deepgpu:0.1.0
datasets:
  trainval2007: '{PascalVOC2007-trainval_ID}'
  trainval2012: '{PascalVOC2012-trainval_ID}'
  test2007: '{PascalVOC2007-test_ID}'
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

## Performance
PASCAL VOC2007 Test

| Model | Original | ChainerCV (weight conversion) | ChainerCV (train) |
|:-:|:-:|:-:|:-:|
| SSD300 | 77.5 % [2] | 77.8 % | 77.5 % / 77.6 % (4 GPUs) |
| SSD512 | 79.5 % [2] | 79.7 % | 80.1 % * / 80.5 % (4 GPUs) |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

\*: We set batchsize to 24 because of memory limitation. The original paper used 32.

## References
1. Wei Liu et al. "SSD: Single shot multibox detector" ECCV 2016.
2. Cheng-Yang Fu et al. "[DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)" arXiv 2017.
