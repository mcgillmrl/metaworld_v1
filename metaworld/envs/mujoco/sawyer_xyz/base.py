import abc
import copy
import os

from gym.spaces import Discrete
import mujoco_py
import numpy as np

from xml.etree import ElementTree as ET

from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat, quat_create, quat_mul
OBS_TYPE = [
    'plain', 'with_goal_id', 'with_goal_and_id', 'with_goal',
    'with_goal_init_obs', 'just_goal'
]


class SawyerMocapBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self,
                 model_name,
                 intervention_id=None,
                 experiment_id=None,
                 frame_skip=5,
                 phase='train',
                 apply_mod=True,
                 remote_render=False,
                 view_setting=[-70, -10, 1.66]):
        self.frame_skip = frame_skip
        self.view_setting = view_setting
        if apply_mod:
            self.apply_env_modifications(model_name, intervention_id,
                                         experiment_id, phase)
        else:
            MujocoEnv.__init__(self,
                               model_name,
                               frame_skip=self.frame_skip,
                               view_setting=self.view_setting)
        self.reset_mocap_welds()

    def apply_env_modifications(self, model_name, intervention_id,
                                experiment_id, phase):
        xmldoc = ET.parse(model_name)
        root = xmldoc.getroot()

        for comp in root.iter('compiler'):
            # Directory hierarchy is different for randomized xmls
            comp.set('meshdir', '../../../meshes')
            comp.set('texturedir', '../../../textures')

        if phase == 'train':
            table_col_tag = int(intervention_id[0])
            goal_col_tag = int(intervention_id[1])
            sky_col_tag = int(intervention_id[2])
            # Make sure these match the interventions tags inside `env_dict.py`
            for geom in root.iter('geom'):
                if geom.get('name') == 'tableTop':
                    if table_col_tag == 0:
                        # Table: Dark Wood
                        geom.set('material', 'darkwood')

                    elif table_col_tag == 1:
                        # Table: Marble
                        geom.set('material', 'marble')

                    elif table_col_tag == 2:
                        # Table: Blue
                        geom.set('material', 'navy_blue')

                    elif table_col_tag == 3:
                        # Table: Tennis
                        geom.set('material', 'tennis')

                    elif table_col_tag == 4:
                        # Table: Wood
                        geom.set('material', 'wood')

                    elif table_col_tag == 5:
                        # Table: LightWood
                        geom.set('material', 'light_wood_v3')

                    elif table_col_tag == 6:
                        # Table: LightWoodV2
                        geom.set('material', 'light_wood_v2')

            for site in root.iter('site'):
                if site.get('name') == 'goal_reach' or site.get(
                        'name') == 'goal_push' or site.get(
                            'name') == 'goal_pick_place':
                    if goal_col_tag == 0:
                        # Goal Red
                        site.set('rgba', '0.8 0 0 1')

                    elif goal_col_tag == 1:
                        # Goal Yellow
                        site.set('rgba', '0.8 1 0.2 1')

                    elif goal_col_tag == 2:
                        # Goal Blue
                        site.set('rgba', '0.1 0.1 1 1')

                    elif goal_col_tag == 3:
                        # Goal Pink
                        site.set('rgba', '1 0.1 0.9 1')

            # Change the background
            for asset in root.iter('asset'):
                for texture in asset.iter('texture'):
                    if texture.get('name') == 'sky':
                        if sky_col_tag == 1:
                            # Red stary sky
                            texture.set('rgb1', '.8 .1 .4')
                            texture.set('rgb2', '0 0 0')
                            texture.set('width', '800')
                            texture.set('height', '800')
                            texture.set('mark', 'random')
                            texture.set('markrgb', '1 1 1')

                        elif sky_col_tag == 2:
                            # Orange stary sky
                            texture.set('rgb1', '.8 .5 .1')
                            texture.set('rgb2', '0 0 0')
                            texture.set('width', '800')
                            texture.set('height', '800')
                            texture.set('mark', 'random')
                            texture.set('markrgb', '1 1 1')

                        elif sky_col_tag == 3:
                            # Yellow stary sky
                            texture.set('rgb1', '1 1 .4')
                            texture.set('rgb2', '0 0 0')
                            texture.set('width', '800')
                            texture.set('height', '800')
                            texture.set('mark', 'random')
                            texture.set('markrgb', '1 1 1')

                        elif sky_col_tag == 4:
                            # Pink stary sky
                            texture.set('rgb1', '1 .5 1')
                            texture.set('rgb2', '0 0 0')
                            texture.set('width', '800')
                            texture.set('height', '800')
                            texture.set('mark', 'random')
                            texture.set('markrgb', '1 1 1')

                        elif sky_col_tag == 5:
                            # Amber stary sky
                            texture.set('rgb1', '1 .6 .4')
                            texture.set('rgb2', '0 0 0')
                            texture.set('width', '800')
                            texture.set('height', '800')
                            texture.set('mark', 'random')
                            texture.set('markrgb', '1 1 1')

                        elif sky_col_tag == 6:
                            # Black stary sky
                            texture.set('rgb1', '0 0 0')
                            texture.set('rgb2', '0 0 0')
                            texture.set('width', '800')
                            texture.set('height', '800')
                            texture.set('mark', 'random')
                            texture.set('markrgb', '1 1 1')

        else:
            # Evaluation phase
            # Change the background
            for asset in root.iter('asset'):
                for texture in asset.iter('texture'):
                    # DMSuite(blue) stary sky
                    if texture.get('name') == 'sky':
                        texture.set('rgb1', '.4 .6 .8')
                        texture.set('rgb2', '0 0 0')
                        texture.set('width', '800')
                        texture.set('height', '800')
                        texture.set('mark', 'random')
                        texture.set('markrgb', '1 1 1')

            # for geom in root.iter('geom'):
            #     if geom.get('name') == 'tableTop':
            #         # Table: Robot
            #         geom.set('material', 'robot')
            # for site in root.iter('site'):
            #     if site.get('name') == 'goal_reach':
            #         # Goal Purple
            #         site.set('rgba', '0.6 0.3 1 1')

            # if intervention_id == 0:
            #     for geom in root.iter('geom'):
            #         if geom.get('name') == 'tableTop':
            #             # Table: Granite
            #             geom.set('material', 'navy_blue')
            #     for site in root.iter('site'):
            #         if site.get('name') == 'goal_reach':
            #             # Goal Red
            #             site.set('rgba', '0.8 0 0 1')
            # else:
            #     raise ValueError('Invalid eval intervention id tag.')

        modxmlpath = model_name.split('sawyer_xyz')[0]
        modxmlpath += 'sawyer_xyz_randomized' + experiment_id + "/"

        if not os.path.exists(modxmlpath):
            os.makedirs(modxmlpath)
        tmppath = modxmlpath + 'phase_' + phase + '.xml'
        xmldoc.write(tmppath)
        MujocoEnv.__init__(self,
                           tmppath,
                           frame_skip=self.frame_skip,
                           view_setting=self.view_setting)

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_gripper_pos(self):
        return np.array([self.data.qpos[7]])

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['sim']
        del state['data']
        mjb = self.model.get_mjb()
        return {'state': state, 'mjb': mjb, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state['state']
        self.model = mujoco_py.load_model_from_mjb(state['mjb'])
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    def __init__(self,
                 *args,
                 hand_low=(-0.2, 0.55, 0.05),
                 hand_high=(0.2, 0.75, 0.3),
                 mocap_low=None,
                 mocap_high=None,
                 action_scale=2. / 100,
                 action_rot_scale=1.,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_xyz_action_rot(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        rot_axis = action[4:] / np.linalg.norm(action[4:])
        action[3] = action[3] * self.action_rot_scale
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        # replace this with learned rotation
        quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi),
                        quat_create(np.array(rot_axis), action[3]))
        self.data.set_mocap_quat('mocap', quat)
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), np.sin(action[3]/2)*rot_axis[0], np.sin(action[3]/2)*rot_axis[1], np.sin(action[3]/2)*rot_axis[2]]))
        # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_xyz_action_rotz(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        zangle_delta = action[3] * self.action_rot_scale
        new_mocap_zangle = quat_to_zangle(
            self.data.mocap_quat[0]) + zangle_delta

        # new_mocap_zangle = action[3]
        new_mocap_zangle = np.clip(
            new_mocap_zangle,
            -3.0,
            3.0,
        )
        if new_mocap_zangle < 0:
            new_mocap_zangle += 2 * np.pi
        self.data.set_mocap_quat('mocap', zangle_to_quat(new_mocap_zangle))

    def set_xy_action(self, xy_action, fixed_z):
        delta_z = fixed_z - self.data.mocap_pos[0, 2]
        xyz_action = np.hstack((xy_action, delta_z))
        self.set_xyz_action(xyz_action)

    def discretize_goal_space(self, goals=None):
        if goals is None:
            self.discrete_goals = [self.default_goal]
        else:
            assert len(goals) >= 1
            self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    # Belows are methods for using the new wrappers.
    # `sample_goals` is implmented across the sawyer_xyz
    # as sampling from the task lists. This will be done
    # with the new `discrete_goals`. After all the algorithms
    # conform to this API (i.e. using the new wrapper), we can
    # just remove the underscore in all method signature.
    def sample_goals_(self, batch_size):
        if self.discrete_goal_space is not None:
            return [
                self.discrete_goal_space.sample() for _ in range(batch_size)
            ]
        else:
            return [self.goal_space.sample() for _ in range(batch_size)]

    def set_goal_(self, goal):
        if self.discrete_goal_space is not None:
            self.active_discrete_goal = goal
            self.goal = self.discrete_goals[goal]
            self._state_goal_idx = np.zeros(len(self.discrete_goals))
            self._state_goal_idx[goal] = 1.
        else:
            self.goal = goal

    def set_init_config(self, config):
        assert isinstance(config, dict)
        for key, val in config.items():
            self.init_config[key] = val

    '''
    Functions that are copied and pasted everywhere and seems
    to be not used.
    '''

    def sample_goals(self, batch_size):
        '''Note: should be replaced by sample_goals_ if not used'''
        # Required by HER-TD3
        goals = self.sample_goals_(batch_size)
        if self.discrete_goal_space is not None:
            goals = [self.discrete_goal_space[g].copy() for g in goals]
        return {
            'state_desired_goal': goals,
        }

    def sample_task(self):
        '''Note: this can be replaced by sample_goal_(batch_size=1)'''
        goal = self.sample_goals_(1)
        if self.discrete_goal_space is not None:
            return self.discrete_goals[goal]
        else:
            return goal

    def _set_obj_xyz_quat(self, pos, angle):
        quat = quat_create(np.array([0, 0, .1]), angle)
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)
