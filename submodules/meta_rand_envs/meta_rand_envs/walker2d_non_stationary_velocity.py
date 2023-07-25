import colorsys
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from meta_rand_envs.base import NonStationaryGoalVelocityEnv


class Walker2DNonStationaryVelocityEnv(NonStationaryGoalVelocityEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.termination_possible = kwargs.get('termination_possible', False)
        NonStationaryGoalVelocityEnv.__init__(self, *args, **kwargs)
        MujocoEnv.__init__(self, 'walker2d.xml', 4)
        utils.EzPickle.__init__(self)
        # should actually go into NonStationaryGoalVelocityEnv, breaks abstraction
        self._init_geom_rgba = self.model.geom_rgba.copy()

        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks

    def step(self, action):
        self.check_env_change()

        posbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        forward_vel = (posafter - posbefore) / self.dt
        reward_run = -1.0 * abs(forward_vel - self.active_task)
        reward_alive = alive_bonus
        reward_ctrl = - 1e-3 * np.square(action).sum()
        reward = reward_run + reward_alive + reward_ctrl
        if self.termination_possible:
            done = not (height > 0.8 and height < 2.0 and
                        ang > -1.0 and ang < 1.0)
        else:
            done = False
        ob = self._get_obs()
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=0, specification=self.active_task))

    # from pearl
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel().astype(np.float32).flatten()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task['velocity']
        self.reset_change_points()
        self.recolor()
        self.steps = 0
        self.reset()
