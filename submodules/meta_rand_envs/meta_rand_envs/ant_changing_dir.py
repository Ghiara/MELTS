import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils


class AntChangingDirEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=0, hfield_mode="gentle", change_mode="location", change_prob=0.005, change_steps=100, termination_possible=True, positive_change_point_basis=20, negative_change_point_basis=-20, change_point_interval=10, state_reconstruction_clip=1000):
        self.change_mode = change_mode
        self.change_prob = change_prob
        self.change_steps = change_steps
        self.termination_possible = termination_possible
        self.changed = False
        self.steps = 0
        self.goal_direction_start = 1
        self.goal_direction = self.goal_direction_start
        self.positive_change_point_basis = positive_change_point_basis
        self.negative_change_point_basis = negative_change_point_basis
        self.change_point_interval = change_point_interval
        self.positive_change_point = np.random.randint(self.positive_change_point_basis,
                                                       self.positive_change_point_basis + self.change_point_interval)
        self.negative_change_point = np.random.randint(self.negative_change_point_basis - self.change_point_interval,
                                                       self.negative_change_point_basis)
        RandomEnv.__init__(self, log_scale_limit, 'ant.xml', 5, hfield_mode=hfield_mode, rand_params=[])
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, action):
        # with some probability change goal direction
        if self.change_mode == "time":
            prob = np.random.uniform(0, 1)
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize and not self.changed:
                self.change_goal_direction()
            if prob < self.change_prob and self.steps > self.change_steps and not self.initialize and self.changed:
                self.change_goal_direction_reset()

        # change direction at some position in the world
        if self.change_mode == "location":
            if self.get_body_com("torso")[
                0] > self.positive_change_point and not self.initialize and self.goal_direction == 1:
                self.change_goal_direction()
                self.positive_change_point = np.random.randint(self.positive_change_point_basis,
                                                               self.positive_change_point_basis + self.change_point_interval)
            if self.get_body_com("torso")[
                0] < self.negative_change_point and not self.initialize and self.goal_direction == -1:
                self.change_goal_direction()
                self.negative_change_point = np.random.randint(
                    self.negative_change_point_basis - self.change_point_interval,
                    self.negative_change_point_basis)

        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt * self.goal_direction
        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - 0 * ctrl_cost - 0 * contact_cost + 0 * survive_reward
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= 0.30 and state[2] <= 10.0 # original gym values: 0.2 and 1.0
            done = not notdone
        else:
            done = False
        self.steps += 1
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            true_task=self.goal_direction)


    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ]).astype(np.float32).flatten() #.astype(np.float32).flatten() added to make compatible

    def reset_model(self):
        # change related
        self.change_goal_direction_reset()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.type = 1
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -20

    def change_goal_direction(self):
        self.goal_direction = -1 * self.goal_direction

        self.recolor()
        self.changed = True
        self.steps = 0

    def change_goal_direction_reset(self):
        # reset changes
        self.goal_direction = self.goal_direction_start

        self.recolor()
        self.changed = False
        self.steps = 0

    def recolor(self):
        if self.goal_direction == 1:
            self.model.geom_rgba[:] = self._init_geom_rgba.copy()
        elif self.goal_direction == -1:
            geom_rgba = self._init_geom_rgba.copy()
            geom_rgba[1:, :3] = np.array([1, 0, 0])
            self.model.geom_rgba[:] = geom_rgba
        else:
            raise NotImplementedError

    def sample_tasks(self, num_tasks):
        directions = np.array([1])
        tasks = [{'direction': direction} for direction in directions]
        return tasks


if __name__ == "__main__":

    env = AntChangingDirEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(2000):
            env.render()
            env.step(env.action_space.sample())  # take a random action

