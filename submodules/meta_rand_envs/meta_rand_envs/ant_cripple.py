import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import numpy as np

class AntCrippleEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0, mode="gentle", change_prob=0.01):
        self.change_prob = change_prob
        self.cripple_mask = None
        self.disabled = False
        self.steps = 0
        RandomEnv.__init__(self, log_scale_limit, 'ant.xml', 5, hfield_mode=mode, rand_params=['body_mass', 'dof_damping', 'body_inertia', 'geom_friction'])
        utils.EzPickle.__init__(self)
        self.cripple_mask = np.ones(self.action_space.shape)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()


    def _step(self, a):
        # with some probability cripple
        prob = np.random.uniform(0, 1)
        if prob < self.change_prob and self.steps > 100 and not self.initialize and not self.disabled:
            self.cripple()

        # use cripple mask
        if self.cripple_mask is not None:
            a = self.cripple_mask * a
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
        # cripple related
        self.cripple_reset()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def cripple(self, value=None):
        # original meta learning case: Pick which leg to remove (0 1 2 are train... 3 is test)
        # testwise multi-task case: Pick which leg to remove (0 1 2 3 are train and test)
        self.crippled_leg = value if value is not None else np.random.randint(0, 4)

        # Pick which actuators to disable
        self.cripple_mask = np.ones(self.action_space.shape)
        if self.crippled_leg == 0:
            self.cripple_mask[2] = 0
            self.cripple_mask[3] = 0
        elif self.crippled_leg == 1:
            self.cripple_mask[4] = 0
            self.cripple_mask[5] = 0
        elif self.crippled_leg == 2:
            self.cripple_mask[6] = 0
            self.cripple_mask[7] = 0
        elif self.crippled_leg == 3:
            self.cripple_mask[0] = 0
            self.cripple_mask[1] = 0

        # Make the removed leg look red
        geom_rgba = self._init_geom_rgba.copy()
        if self.crippled_leg == 0:
            geom_rgba[3, :3] = np.array([1, 0, 0])
            geom_rgba[4, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 1:
            geom_rgba[6, :3] = np.array([1, 0, 0])
            geom_rgba[7, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 2:
            geom_rgba[9, :3] = np.array([1, 0, 0])
            geom_rgba[10, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 3:
            geom_rgba[12, :3] = np.array([1, 0, 0])
            geom_rgba[13, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba

        # Make the removed leg not affect anything
        temp_size = self._init_geom_size.copy()
        temp_pos = self._init_geom_pos.copy()

        if self.crippled_leg == 0:
            # Top half
            temp_size[3, 0] = temp_size[3, 0] / 2
            temp_size[3, 1] = temp_size[3, 1] / 2
            # Bottom half
            temp_size[4, 0] = temp_size[4, 0] / 2
            temp_size[4, 1] = temp_size[4, 1] / 2
            temp_pos[4, :] = temp_pos[3, :]

        elif self.crippled_leg == 1:
            # Top half
            temp_size[6, 0] = temp_size[6, 0] / 2
            temp_size[6, 1] = temp_size[6, 1] / 2
            # Bottom half
            temp_size[7, 0] = temp_size[7, 0] / 2
            temp_size[7, 1] = temp_size[7, 1] / 2
            temp_pos[7, :] = temp_pos[6, :]

        elif self.crippled_leg == 2:
            # Top half
            temp_size[9, 0] = temp_size[9, 0] / 2
            temp_size[9, 1] = temp_size[9, 1] / 2
            # Bottom half
            temp_size[10, 0] = temp_size[10, 0] / 2
            temp_size[10, 1] = temp_size[10, 1] / 2
            temp_pos[10, :] = temp_pos[9, :]

        elif self.crippled_leg == 3:
            # Top half
            temp_size[12, 0] = temp_size[12, 0] / 2
            temp_size[12, 1] = temp_size[12, 1] / 2
            # Bottom half
            temp_size[13, 0] = temp_size[13, 0] / 2
            temp_size[13, 1] = temp_size[13, 1] / 2
            temp_pos[13, :] = temp_pos[12, :]

        self.model.geom_size[:] = temp_size
        self.model.geom_pos[:] = temp_pos
        self.disabled = True

    def cripple_reset(self):
        self.cripple_mask = np.ones(self.action_space.shape)
        self.disabled = False
        self.steps = 0
        self.model.geom_rgba[:] = self._init_geom_rgba
        self.model.geom_size[:] = self._init_geom_size
        self.model.geom_pos[:] = self._init_geom_pos


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

