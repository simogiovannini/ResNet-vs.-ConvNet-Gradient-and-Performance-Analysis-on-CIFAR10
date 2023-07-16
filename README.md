# DLA-lab1 (README work in progress)
Deep Learning Applications laboratory on CNNs


## Models
In `models/` we define the three architectures that were tested during this laboratory.

### MLP
A simple 4-layers MultiLayerPerceptron is implemented in `models/mlp.py`.

### Convolutional Network
`models/convnet.py` contains the implementation of ResNet34 described in this [paper](https://arxiv.org/abs/1512.03385).
To compare the effectiveness of residual connections, the user can instantiate the architecture choosing to add skip connections or not through the `skip_connection` parameter specified in the constructor.

If `skip_connection == True` the architecture will contain residual connections otherwise it will not.

### ResNet50

To compare our implementation with a bigger architecture we copied from the web the implementation of ResNet50 and copied it in `models/resnet.py`.


## RDataset: CIFAR10

[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) was used to train and test our models.

The only detail that has to be mentioned is that, while implementing `utils/data_loader.py`, we upscaled CIFAR's images from 32x32 to 128x128 in order to produce more detailed CAMs (see exercise 2.3).


## Exercise 1.1: Comparing Architectures

In `3_1.py` we applied Reinforce algorithm on gymnasium's [Lunar Lander environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

The algorithm was tested with 3 different values for the `temperature` parameter setting the number of episodes to 30K:
- `temperature = 0.0` (represented in green)
- `temperature = 0.8` (represemted in pink)
- `temperature = (episode/num_episodes) * 0.9` that increases linearly during learning (represemted in yellow)

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/423810aa-9660-4495-ba7f-91b1743d71e3)

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/078c4a7a-7cc1-4ea7-b4be-5d471b751d5e)

The first graph represents the average reward collected by the agent during the last 10 episodes while the second represents the number of steps of the last episode.

Unexpectedly the best performance are provided by the version that always samples from the actions' distribution. The other two runs does not converge at all and it's clear also from the second graph.

The best model reaches an average reward of around 130.


## Exercise 3.3: Proximal Policy Optimization
In `3_3.py` we applied PPO from [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) to Lunar Lander.

The algorithm was tested varying the number of timesteps using these set of values: `[2'500, 5'000, 10'000, 25'000, 50'000, 100'000, 150'000, 200'000, 250'000, 500'000, 1'000'000, 2'500'000, 5'000'000]`.

![image](https://github.com/simogiovannini/DLA-lab3/assets/53260220/76594ca9-2940-43cb-841e-3e74d0031de7)

In this graph the average reward of PPO is represented with the blue line and it's compared to the green line seen before. It's clear how PPO overperforms Reinforce both in time and in reached reward.
15 minutes of training are enough to reach way better performances.


## Requirements
You can use the `requirements.txt` file to create the conda environment to run the code in this repository.
