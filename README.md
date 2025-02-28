# Model Agnostic Meta Learning (MAML) applied to Reinforcement Learning (RL)
The purpose of the project is an easy to apply/replicate adaptation of MAML to the field of Reinforcement Learning.

## How to install
To use with your favourite Package/Environment manager with pythong 3.9.20 create a dedicated environment.
Install the necessary dependencies using:
```
pip install -r requirements.txt
```

## Troubleshooting MuJoCo visualisation
MuJoCo rendering can be tough to make working, so the following are recommendation that works on my computer (Linux) in case you have trouble making it work:
```
 1. Ensure you are using X instead of Wayland. Set WaylandEnable=false in the file /etc/gdm3/custom.conf 
 2. In the terminal execute the following commands:
    export MUJOCO_GL=egl
    export DISPLAY=:1
    export MUJOCO_GL="glfw"
```
## Training a Policy Gradient using MAML
The only available for multitasking, HalfCheetahDir is HalfCheetah where each task correspond to a different direction. It is launched using the following line:
```
pip install -e 
python easy_maml/scripts/run_main.py --env_name HalfCheetahDir -rtg --discount 0.95 --maml --exp_name HalfCheetah_MAML_PG
```

## Testing a Policy Gradient 
During the test we load the trained model and we will perform a few gradient pass to get it to specialise on the task.
```
pip install -e 
python easy_maml/scripts/run_main.py --env_name HalfCheetahDir -n 10 -rtg --discount 0.95 -lr 0.01 --test --video_log_freq 9 --exp_name HalfCheetah_PG
```
## Acknowledgements
This project is an implementation of the following paper:
Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.
reference: [[ArXiv](https://arxiv.org/abs/1703.03400)]

The implementation of the HalfCheetahDir wrapper and environment has been reproduced from the following implementation:
Tristan Deleu. Model-Agnostic Meta-Learning for Reinforcement Learning in PyTorch
reference: [[github](https://github.com/tristandeleu/pytorch-maml-rl/maml_rl/envs)] 

