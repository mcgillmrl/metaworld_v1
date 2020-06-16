import os.path as osp
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py

class JacoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='/home/melissa/Workspace/metaworld/metaworld/envs/assets/jaco_xyz/jaco_reach_push_pick_and_place.xml',
                 ):
        #import pdb;pdb.set_trace()
        utils.EzPickle.__init__(**locals())
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self._reset_noise_scale = 0.01
        self.block_gripper = False

        self._env_setup()

    def _env_setup(self):
        self.reset_mocap_welds()
        self.sim.forward()

        # Move end effector into position.
        # @Melissa: TODO: You need to figure out what to set these two values to
        gripper_target = np.array([-0.498, 0.005, -0.431])
        # gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('mocap', gripper_target)
        # self.sim.data.set_mocap_quat('mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def step(self, action):
        action = action.copy()

        if len(action) == 3:
            # @Melissa: This only happens during init? I'm not sure how to fix it cleanly
            action = np.zeros(9)

        pos_ctrl, gripper_ctrl = action[:3], action[3:]

        # @Melissa: You'll need to play around with these two as well
        # pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [-0.5, 0.5, -0.5, 0.5]  # fixed rotation of the end effector, expressed as a quaternion
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        self.ctrl_set_action(gripper_ctrl)
        self.mocap_set_action(action)
        self.sim.step()

        observation = self._get_obs()
        reward = 0
        done = False
        return observation, reward, done, {}

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def ctrl_set_action(self, action):
        """For torque actuators it copies the action into mujoco ctrl field.
        For position actuators it sets the target relative to the current qpos.
        """

        # @Melissa: This needs to be changed because you have 6DOF on the EndEffector, but this only does the last three
        for i in (-1, -2, -3):
            self.sim.data.ctrl[i] =  action[i]

    def mocap_set_action(self, action):
        """The action controls the robot using mocaps. Specifically, bodies
        on the robot (for example the gripper wrist) is controlled with
        mocap bodies. In this case the action is the desired difference
        in position and orientation (quaternion), in world coordinates,
        of the of the target body. The mocap is positioned relative to
        the target body according to the delta, and the MuJoCo equality
        constraint optimizer tries to center the welded body on the mocap.
        """
        # @Melissa: Action = 3DOF Cartesian Position Delta + Quaternion
        if self.sim.model.nmocap > 0:
            action, _ = np.split(action, (self.sim.model.nmocap * 7, ))
            action = action.reshape(self.sim.model.nmocap, 7)

            pos_delta = action[:, :3]
            quat_delta = action[:, 3:]

            self.reset_mocap2body_xpos()
            self.sim.data.mocap_pos[:] = self.sim.data.mocap_pos + pos_delta
            self.sim.data.mocap_quat[:] = self.sim.data.mocap_quat + quat_delta


    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation.
        """
        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        self.sim.forward()


    def reset_mocap2body_xpos(self):
        """Resets the position and orientation of the mocap bodies to the same
        values as the bodies they're welded to.
        """

        if (self.sim.model.eq_type is None or
            self.sim.model.eq_obj1id is None or
            self.sim.model.eq_obj2id is None):
            return
        for eq_type, obj1_id, obj2_id in zip(self.sim.model.eq_type,
                                            self.sim.model.eq_obj1id,
                                            self.sim.model.eq_obj2id):
            if eq_type != mujoco_py.const.EQ_WELD:
                continue

            mocap_id = self.sim.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                # obj1 is the mocap, obj2 is the welded body
                body_idx = obj2_id
            else:
                # obj2 is the mocap, obj1 is the welded body
                mocap_id = self.sim.model.body_mocapid[obj2_id]
                body_idx = obj1_id

            assert (mocap_id != -1)
            self.sim.data.mocap_pos[mocap_id][:] = self.sim.data.body_xpos[body_idx]
            self.sim.data.mocap_quat[mocap_id][:] = self.sim.data.body_xquat[body_idx]