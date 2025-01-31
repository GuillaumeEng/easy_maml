from gym.envs.registration import load
from gym.wrappers import TimeLimit

from easy_maml.environment.utils.normalized_env import NormalizedActionWrapper

def mujoco_wrapper(entry_point, **kwargs):
    normalization_scale = kwargs.pop('normalization_scale', 1.)

    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    

    # Normalization wrapper
    env = NormalizedActionWrapper(env, scale=normalization_scale)

    return env