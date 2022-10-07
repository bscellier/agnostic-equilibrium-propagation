# Agnostic Equilibrium Propagation

This repository contains code to produce simulations of [Agnostic Equilibrium Propagation](https://arxiv.org/abs/2205.15021) on Hopfield-like networks.

The code uses PyTorch, TorchVision and TensorBoard.

## Getting started
* Download the code from GitHub:
```bash
git clone https://github.com/bscellier/agnostic-equilibrium-propagation
cd agnostic-equilibrium-propagation
```
* To train a Hopfield-like network (with 1 hidden layer by default) on MNIST with Agnostic Eqprop, run the command:
``` bash
python run.py --dataset='MNIST' --architecture='1h' --method='centered' --verbose
```
* To run Tensorboard, use the command:
``` bash
tensorboard --logdir=runs/MNIST/1h/
```