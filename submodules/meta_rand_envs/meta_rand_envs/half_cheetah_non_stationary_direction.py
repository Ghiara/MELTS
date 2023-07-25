import colorsys
import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from meta_rand_envs.base import NonStationaryGoalDirectionEnv


class HalfCheetahNonStationaryDirectionEnv(NonStationaryGoalDirectionEnv, MujocoEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.termination_possible = kwargs.get('termination_possible', False)
        NonStationaryGoalDirectionEnv.__init__(self, *args, **kwargs)
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
        # should actually go into NonStationaryGoalDirectionEnv, breaks abstraction
        self._init_geom_rgba = self.model.geom_rgba.copy()

        self.train_tasks = self.sample_tasks(kwargs['n_train_tasks'])
        self.test_tasks = self.sample_tasks(kwargs['n_eval_tasks'])
        self.tasks = self.train_tasks + self.test_tasks

    def step(self, action):
        self.check_env_change()

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_run = (xposafter - xposbefore) / self.dt * self.active_task
        reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
        reward = reward_ctrl * 1.0 + reward_run
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=0, specification=self.active_task), direction=(xposafter - xposbefore))

    # from pearl
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.get_body_com("torso").flat,
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task['direction']
        self.reset_change_points()
        self.recolor()
        self.steps = 0
        self.reset()
