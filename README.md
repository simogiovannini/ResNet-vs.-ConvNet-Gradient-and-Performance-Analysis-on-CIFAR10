# DLA-lab1 (README work in progress)
Deep Learning Applications laboratory on CNNs


## Models
In `models/` we define the three architectures that were tested during this laboratory.

### MLP
A simple 4-layers MultiLayerPerceptron is implemented in `models/mlp.py`.

### Convolutional/Residual Network
`models/convnet.py` contains the implementation of ResNet34 described in this [paper](https://arxiv.org/abs/1512.03385).
To compare the effectiveness of residual connections, the user can instantiate the architecture choosing to add skip connections or not through the `skip_connection` parameter specified in the constructor.

If `skip_connection == True` the architecture will contain residual connections otherwise it will not.

### ResNet50

To compare our implementation with a bigger architecture we copied from the web the implementation of ResNet50 and copied it in `models/resnet.py`.


## RDataset: CIFAR10

[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) was used to train and test our models.

The only detail that has to be mentioned is that, while implementing `utils/data_loader.py`, we upscaled CIFAR's images from 32x32 to 128x128 in order to produce more detailed CAMs (see exercise 2.3).


## Exercise 1.1: Comparing Architectures

`1_1.py` contains a comparison between four architectures on CIFAR10:
- MultiLayerPerceptron
- ResNet34
- ConvNet34 (the same architecture as ResNet34 but without residual connections)
- ResNet50

Each model was trained for the same number of epochs, using the same train/validation split. Here we analyze the results:

![image](https://github.com/simogiovannini/DLA-lab1/assets/53260220/18ba47e7-1d1c-4084-866e-67e4a9c246fd)

![image](https://github.com/simogiovannini/DLA-lab1/assets/53260220/487c0372-9911-42ab-b0c2-47ab0ba28ee2)

MLP is completely uneffective on this task while all the CNNs share the same behaviour: they work nicely but the rapidly overfit. We can see it from the growth of validation loss over time.

There is no particular difference between ResNet34 and ResNet50.


## Exercise 3.3: Proximal Policy Optimization
In `3_3.py` we applied PPO from [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) to Lunar Lander.

The algorithm was tested varying the number of timesteps using these set of values: `[2'500, 5'000, 10'000, 25'000, 50'000, 100'000, 150'000, 200'000, 250'000, 500'000, 1'000'000, 2'500'000, 5'000'000]`.

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/76594ca9-2940-43cb-841e-3e74d0031de7)

In this graph the average reward of PPO is represented with the blue line and it's compared to the green line seen before. It's clear how PPO overperforms Reinforce both in time and in reached reward.
15 minutes of training are enough to reach way better performances.


## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
