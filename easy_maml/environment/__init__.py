from gym.envs.registration import register


register(
    'HalfCheetahDir',
    entry_point='easy_maml.environment.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'easy_maml.environment.half_cheetah:HalfCheetahDirEnv',},
    max_episode_steps=1000
)