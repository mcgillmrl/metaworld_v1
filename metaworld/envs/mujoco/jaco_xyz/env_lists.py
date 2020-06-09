'''
This file provide lists of environment for multitask learning.
'''

from metaworld.envs.mujoco.jaco_xyz.sawyer_reach_push_pick_place import JacoReachPushPickPlaceEnv


# jaco tasks
JACO_LIST = [
    JacoReachPushPickPlaceEnv
]


def verify_env_list_space(env_list):
    '''
    This method verifies the action_space and observation_space
    of all environments in env_list are the same.
    '''
    prev_action_space = None
    prev_obs_space = None
    for env_cls in env_list:
        env = env_cls()
        if prev_action_space is None or prev_obs_space is None:
            prev_action_space = env.action_space
            prev_obs_space = env.observation_space
            continue
        assert env.action_space.shape == prev_action_space.shape,\
            '{}, {}, {}'.format(env, env.action_space.shape, prev_action_space)
        assert env.observation_space.shape == prev_obs_space.shape,\
            '{}, {}, {}'.format(env, env.observation_space.shape, prev_obs_space)
        prev_action_space = env.action_space
        prev_obs_space = env.observation_space
