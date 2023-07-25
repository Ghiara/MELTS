import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import colorsys


class Walker2DChangingRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', 'location')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 100)
        self.termination_possible = kwargs.get('termination_possible', False)
        self.steps = 0
        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.task_min_velocity = kwargs.get('task_min_velocity', 0.0)
        self.task_max_velocity = kwargs.get('task_max_velocity', 1.0)

        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        RandomEnv.__init__(self, kwargs.get('log_scale_limit', 0), 'walker2d.xml', 5, hfield_mode=kwargs.get('hfield_mode', 'gentle'), rand_params=kwargs.get('rand_params', []))
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, action):
        # with some probability change goal direction
        if self.change_mode == "time":
            prob = np.random.uniform(0, 1)
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize:
                self.change_goal_velocity()

        # change direction at some position in the world
        if self.change_mode == "location":
            if self.get_body_com("torso")[0] > self.positive_change_point and not self.initialize:
                self.change_goal_velocity()
                self.positive_change_point = self.positive_change_point + self.positive_change_point_basis + np.random.random() * self.change_point_interval

            if self.get_body_com("torso")[0] < self.negative_change_point and not self.initialize:
                self.change_goal_velocity()
                self.negative_change_point = self.negative_change_point + self.negative_change_point_basis - np.random.random() * self.change_point_interval
            
        xposbefore = self.sim.data.qpos[0]
        try:
            self.do_simulation(action, self.frame_skip)
        except:
            raise RuntimeError("Simulation error, common error is action = nan")
        xposafter, height, ang = self.sim.data.qpos[0:3]
        ob = self._get_obs()
        reward_run = (xposafter - xposbefore) / self.dt
        alive_bonus = 1.0
        reward_ctrl = -1e-3 * np.sum(np.square(action))
        reward = reward_ctrl * 1.0 + reward_run + alive_bonus
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            done = not (height > 0.8 and height < 2.0 and
                        ang > -1.0 and ang < 1.0)
        else:
            done = False
        self.steps += 1
        specification = self.sim.model.body_mass[1] * self.sim.model.body_inertia[1][0] * self.sim.model.dof_damping[3] * self.sim.model.geom_friction[1][0]
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, true_task=dict(base_task=0, specification=specification))

    '''
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ]).astype(np.float32).flatten() #.astype(np.float32).flatten() added to make compatible
    '''

    # no velocity
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        # reset changepoint
        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        # reset velocity to task starting velocity
        self.set_task(self._task)
        self.recolor()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20
        
    def change_goal_velocity(self):
        if self.meta_mode == 'train':
            self._task = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self._task = np.random.choice(self.test_tasks)

        self.set_task(self._task)
        self.recolor()
        self.steps = 0

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        hue = np.random.random()
        rgb_value_tuple = colorsys.hsv_to_rgb(hue,1,1)
        geom_rgba[1:, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba

    def set_meta_mode(self, mode):
        self.meta_mode = mode


if __name__ == "__main__":

    env = HalfCheetahChangingVelEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(2000):
            env.render()
            env.step(env.action_space.sample())  # take a random action
