from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.jaco_xyz.base import JacoXYZEnv

from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.jaco_xyz.base import OBS_TYPE


class JacoDrawerCloseEnv(JacoXYZEnv):
    def __init__(self,
                 random_init=False,
                 obs_type='plain',
                 goal_low=None,
                 goal_high=None,
                 rotMode='fixed',
                 **kwargs):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.04)
        obj_high = (0.1, 0.9, 0.04)
        JacoXYZEnv.__init__(self,
                            frame_skip=5,
                            action_scale=1. / 100,
                            hand_low=hand_low,
                            hand_high=hand_high,
                            model_name=self.model_name,
                            **kwargs)

        self.init_config = {
            'obj_init_angle': np.array([
                0.3,
            ], dtype=np.float32),
            'obj_init_pos': np.array([-0.07, 0.75, -0.05], dtype=np.float32),
            'hand_init_pos': np.array([0.108, 0.622, 0.5], dtype=np.float32),
        }
        self.goal = np.array([0, 0.9, -0.05])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        if goal_low is None:
            goal_low = self.hand_low

        if goal_high is None:
            goal_high = self.hand_high

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        self.random_init = random_init
        self.max_path_length = 200
        self.rotMode = rotMode
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1, -1, -1]),
                np.array([1, 1, 1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1. / 50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2 * np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi / 2, -np.pi / 2, 0, -1]),
                np.array([1, 1, 1, np.pi / 2, np.pi / 2, np.pi * 2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((
                    self.hand_low,
                    obj_low,
                )),
                np.hstack((
                    self.hand_high,
                    obj_high,
                )),
            )
        elif self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            raise NotImplementedError
        self.reset()

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
        }

    @property
    def model_name(self):
        return get_asset_full_path('jaco_xyz/jaco_drawer.xml')

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        # self.do_simulation([action[-1], -action[-1]])
        # Melissa : Do we just simulate raw actions, or rotationally
        # modified?
        self.do_simulation(action)

        # print('Actions ', action)
        # print('Joint pose :', self.sim.data.qpos)
        # print('Mocap pose :', self.sim.data.get_mocap_pos('mocap'))

        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pullDist = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1
        #info = self._get_info()
        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pullDist <= 0.06)
        }
        info['goal'] = self.goal
        return ob, reward, False, info

    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos = self.data.get_geom_xpos('handle')
        flat_obs = np.concatenate((hand, objPos))
        if self.obs_type == 'with_goal_and_id':
            return np.concatenate(
                [flat_obs, self._state_goal, self._state_goal_idx])
        elif self.obs_type == 'with_goal':
            return np.concatenate([flat_obs, self._state_goal])
        elif self.obs_type == 'plain':
            return np.concatenate([
                flat_obs,
            ])  # TODO ZP do we need the concat?
        else:
            return np.concatenate([flat_obs, self._state_goal_idx])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos = self.data.get_geom_xpos('handle')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (goal[:3])

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos = self.data.get_geom_xpos('handle')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (objPos)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        # qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        if self.random_init:
            # self.obj_init_pos = np.random.uniform(-0.2, 0)
            # self._state_goal = np.squeeze(np.random.uniform(
            #     self.goal_space.low,
            #     np.array(self.data.get_geom_xpos('handle').copy()[1] + 0.05),
            # ))
            obj_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            # self.obj_init_qpos = goal_pos[-1]
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            # goal_pos[1] -= 0.15
            goal_pos[1] -= 0.2
            self._state_goal = goal_pos
        self._set_goal_marker(self._state_goal)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        drawer_cover_pos = self.obj_init_pos.copy()
        drawer_cover_pos[1] += 0.15
        self.sim.model.body_pos[self.model.body_name2id(
            'drawer')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id(
            'drawer_cover')] = drawer_cover_pos
        self.sim.model.site_pos[self.model.site_name2id(
            'goal')] = self._state_goal
        self._set_obj_xyz(-0.2)
        self.curr_path_length = 0
        self.maxDist = np.abs(
            self.data.get_geom_xpos('handle')[1] - self._state_goal[1])
        self.target_reward = 1000 * self.maxDist + 1000 * 2
        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        # Some initial joint configurations that seem reasonable
        # Since controlling with the position controller is a little
        # finicky
        self.sim.data.set_joint_qpos('jaco_joint_1', -4.42138)
        self.sim.data.set_joint_qpos('jaco_joint_2', -7.09531)
        self.sim.data.set_joint_qpos('jaco_joint_3', -9.5786)
        self.sim.data.set_joint_qpos('jaco_joint_4', 7.700)
        self.sim.data.set_joint_qpos('jaco_joint_5', -6.5482)
        self.sim.data.set_joint_qpos('jaco_joint_6', -3.9436)
        self.sim.data.set_joint_qpos('jaco_joint_7', -6.2832)

        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([0, 0, 0.2, 0, 0, 0], self.frame_skip)

        finger1, finger2, finger3 = self.get_joint_pos(
            'jaco_joint_finger_tip_1'), self.get_joint_pos(
                'jaco_joint_finger_tip_2'), self.get_joint_pos(
                    'jaco_joint_finger_tip_3')
        self.init_fingerCOM = (finger1 + finger2 + finger3) / 3

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def get_joint_pos(self, jointName):
        return self.data.get_joint_qpos(jointName)

    def compute_reward(self, actions, obs):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        objPos = obs[3:6]
        # TODO : Adopt for three finger grab
        finger1, finger2, finger3 = self.get_joint_pos(
            'jaco_joint_finger_tip_1'), self.get_joint_pos(
                'jaco_joint_finger_tip_2'), self.get_joint_pos(
                    'jaco_joint_finger_tip_3')
        fingerCOM = (finger1 + finger2 + finger3) / 3

        pullGoal = self._state_goal[1]

        reachDist = np.linalg.norm(objPos - fingerCOM)

        pullDist = np.abs(objPos[1] - pullGoal)

        # reward = -reachDist - pullDist
        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            # pullRew = -pullDist
            pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3))
            pullRew = max(pullRew, 0)
        else:
            pullRew = 0
        reward = -reachDist + pullRew

        return [reward, reachDist, pullDist]

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths=None, logger=None):
        pass
