import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import numpy as np

class AntRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0, mode="gentle", change_prob=0.01):
        self.change_prob = change_prob
        self.changed = False
        self.steps = 0
        RandomEnv.__init__(self, log_scale_limit, 'ant.xml', 5, hfield_mode=mode, rand_params=['body_mass', 'dof_damping', 'body_inertia', 'geom_friction'])
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, a):
        # with some probability change friction
        prob = np.random.uniform(0, 1)
        if prob < self.change_prob and self.steps > 100 and not self.initialize and not self.changed:
            self.change_parameters()

        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0

        done = not notdone
        ob = self._get_obs()
        self.steps += 1
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        # change related
        self.change_parameters_reset()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def change_parameters(self):
        # intermediate change log_scale_limit
        temp_log_scale_limit = self.log_scale_limit
        self.log_scale_limit = 3
        new_params = self.sample_tasks(1)
        self.set_physical_parameters(new_params[0])
        self.log_scale_limit = temp_log_scale_limit
        # recolor
        geom_rgba = self._init_geom_rgba.copy()
        geom_rgba[1:, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba
        self.changed = True

    def change_parameters_reset(self):
        # reset changes
        self.changed = False
        self.steps = 0
        self.set_physical_parameters(self._task)

        self.model.geom_rgba[:] = self._init_geom_rgba.copy()


if __name__ == "__main__":

    env = AntRandParamsEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(2000):
            env.render()
            env.step(env.action_space.sample())  # take a random action

