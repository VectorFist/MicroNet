# LeNet5
A Lenet5 C++ implementation without using any deep learning framework

In this repository, I implemented a simple Lenet5 network with C++, without using any deep learning framework. Four optimization algorithms are implemented: SGD, Adam, AdaGrad, RMSProb. The experimental results of the four optimization algorithms are as follows. It can be seen that Adam has achieved the best test accuracy 99.15% (99.05% in Lecun's origin paper [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)). This also reflects the superiority of the Adam algorithm, which is less sensitive to the choice of optimizing superparameters.

## Requirements
- Python3
- NumPy
- [Tensorflow >= 1.3](https://github.com/tensorflow/tensorflow)

## Usage
**Step 1.** 
Clone this repository with ``git``.

```
$ git clone https://github.com/VectorFist/CapsNet.git
$ cd CapsNet
```

**Step 2.** 
Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), extract it into ``MNIST_data`` directory.

**Step 3.** 
Start the training:
```
$ python run_capsnet.py
```

**Step 3.** 
Test capsnet model:
```
$ python run_capsnet.py --run_mode=test
```
