# Convolutional neural networks for CIFAR-10 and CIFAR-100 Classification

This is an example of a convolutional neural network (convnet) applied to an image classification task using the CIFAR-10 dataset. The CIFAR datasets can be a good choice for initial experiments with convnets because the size and number of images is small enough to allow typical models to be trained in a reasonable amount of time. However, the classification task is still challenging because natural images are used.

Specifically, there are 50000 color training images of size 32x32 pixels with either 10 class labels (for CIFAR-10).

For CIFAR-10, state of the art methods without data augmentation can achieve similar to human-level classification accuracy of around 94%.

The code consists of three parts: dataset preparation, network and optimizer definition and learning loop, similar to the MNIST example.

This uses the VGG-style network from [here](http://torch.ch/blog/2015/07/30/cifar.html) which is based on the network architecture from the paper from [here](https://arxiv.org/pdf/1409.1556v6.pdf).

No data augmentation is used and the classification accuracy on the CIFAR-10 test set for the VGG-style model should reach approximately 89% after 200 iterations or so.

## Train

### Dataset preperation

https://github.com/abeja-inc/abeja-platform-samples/tree/master/dataset/cifar10

### Training

The training setting sample is here. `{JOB_NAME}` is arbitrarily name that you can define. The dataset ID of `{CIFAR10_ID}` and `{CIFAR10-test_ID}` are described in ABEJA Platform Console.
```
name: {JOB_NAME}
handler: train:handler
image: abeja-inc/deepgpu:0.1.0
datasets:
  train: '{CIFAR10_ID}'
  test: '{CIFAR10-test_ID}'
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
