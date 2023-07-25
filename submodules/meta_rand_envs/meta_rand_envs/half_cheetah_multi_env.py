import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import mujoco_py
import os


class HalfCheetahMixtureEnv(RandomEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', '')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 80)
        self.termination_possible = kwargs.get('termination_possible', False)
        self.steps = 0

        # velocity/position specifications in ranges [from, to]
        self.velocity_x = [1.0, 5.0]
        self.pos_x = [5.0, 25.0]
        self.velocity_y = [2. * np.pi, 4. * np.pi]
        self.pos_y = [np.pi / 6., np.pi / 2.]
        self.velocity_z = [1.5, 3.]

        # self.velocity_x = [1.0, 4.0]
        # self.pos_x = [5.0, 10.0]
        # self.velocity_y = [2. * np.pi, 4. * np.pi]
        # self.pos_y = [np.pi / 5., np.pi / 2.]
        # self.velocity_z = [1.5, 3.]

        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.base_task = 0
        self.task_specification = 1.0
        task_names = ['velocity_forward', 'velocity_backward',
                      'goal_forward', 'goal_backward',
                      'flip_forward',
                      'stand_front', 'stand_back',
                      'jump',
                      'direction_forward', 'direction_backward',
                      'velocity']
        self.task_variants = kwargs.get('task_variants', task_names)
        self.bt2t = {k: self.task_variants.index(k) if k in self.task_variants else -1 for k in task_names}

        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        self.model_path = os.path.join(os.getcwd(), 'submodules', 'meta_rand_envs', 'meta_rand_envs',
                                       'half_cheetah.xml')
        RandomEnv.__init__(self, kwargs.get('log_scale_limit', 0), self.model_path, 5,
                           hfield_mode=kwargs.get('hfield_mode', 'gentle'), rand_params=[])
        utils.EzPickle.__init__(self)

        self._init_geom_rgba = self.model.geom_rgba.copy()

    def _step(self, action):
        # change task after some steps
        if self.change_mode == "time" and not self.initialize:

            if 'current_step' not in self.tasks[self.last_idx].keys():
                self.tasks[self.last_idx]['current_step'] = 0
            self.tasks[self.last_idx]['current_step'] += 1

            if 'changed_task_spec' in self.tasks[self.last_idx].keys():
                self.change_task(self.tasks[self.last_idx]['changed_task_spec'])
            if self.tasks[self.last_idx]['current_step'] % self.change_steps == 0:
                task_spec = np.random.choice(self.train_tasks if self.meta_mode == 'train' else self.test_tasks)
                self.tasks[self.last_idx]['changed_task_spec'] = {
                    'base_task': task_spec['base_task'],
                    'specification': task_spec['specification'],
                    'color': task_spec['color']
                }
                self.change_task(self.tasks[self.last_idx]['changed_task_spec'])
                self.tasks[self.last_idx]['current_step'] = 0

        xposbefore = np.copy(self.sim.data.qpos)
        try:
            self.do_simulation(action, self.frame_skip)
        except:
            raise RuntimeError("Simulation error, common error is action = nan")

        xposafter = np.copy(self.sim.data.qpos)
        xvelafter = np.copy(self.sim.data.qvel)

        ob = self._get_obs()

        if self.base_task in [self.bt2t['velocity_forward'], self.bt2t['velocity_backward']]:  # 'velocity'
            reward_run = - np.abs(xvelafter[0] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['goal_forward'], self.bt2t['goal_backward']]:  # 'goal'
            reward_run = - np.abs(xposafter[0] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['flip_forward']]:  # 'flipping'
            reward_run = - np.abs(xvelafter[2] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['stand_front'], self.bt2t['stand_back']]:  # 'stand_up'
            reward_run = - np.abs(xposafter[2] - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['jump']]:  # 'jump'
            reward_run = - np.abs(np.abs(xvelafter[1]) - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        elif self.base_task in [self.bt2t['direction_forward'], self.bt2t['direction_backward']]:  # 'direction'
            reward_run = (xposafter[0] - xposbefore[0]) / self.dt * self.task_specification
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run

        elif self.base_task in [self.bt2t['velocity']]:
            forward_vel = (xposafter[0] - xposbefore[0]) / self.dt
            reward_run = -1.0 * np.abs(forward_vel - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run
        else:
            raise RuntimeError("bask task not recognized")

        # print(str(self.base_task) + ": " + str(reward))
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= -2.5 and state[2] <= 2.5
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl,
                                      true_task=dict(base_task=self.base_task, specification=self.task_specification))

    # from pearl rlkit
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

        # reset tasks
        self.base_task = self._task['base_task']
        self.task_specification = self._task['specification']
        self.recolor()

        # standard
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_image(self, width=256, height=256, camera_name=None):
        if self.viewer is None or type(self.viewer) != mujoco_py.MjRenderContextOffscreen:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim)
            self.viewer_setup()
            self._viewers['rgb_array'] = self.viewer

        # use sim.render to avoid MJViewer which doesn't seem to work without display
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    def viewer_setup(self):
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = 0

    def change_task(self, spec):
        self.base_task = spec['base_task']
        self.task_specification = spec['specification']
        self._goal = spec['specification']
        self.color = spec['color']
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value = self.color
        geom_rgba[1:, :3] = np.asarray(rgb_value)
        self.model.geom_rgba[:] = geom_rgba

    def sample_tasks(self, num_tasks_list):
        if type(num_tasks_list) != list: num_tasks_list = [num_tasks_list]

        num_base_tasks = len(self.task_variants)
        num_tasks_per_subtask = [int(num_tasks / num_base_tasks) for num_tasks in num_tasks_list]
        num_tasks_per_subtask_cumsum = np.cumsum(num_tasks_per_subtask)

        tasks = [[] for _ in range(len(num_tasks_list))]
        # velocity tasks
        if 'velocity_forward' in self.task_variants:
            velocities = np.linspace(self.velocity_x[0], self.velocity_x[1], num=sum(num_tasks_per_subtask))
            tasks_velocity = [
                {'base_task': self.bt2t['velocity_forward'], 'specification': velocity, 'color': np.array([1, 0, 0])}
                for velocity in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'velocity_backward' in self.task_variants:
            velocities = np.linspace(-self.velocity_x[1], -self.velocity_x[0], num=sum(num_tasks_per_subtask))
            tasks_velocity = [
                {'base_task': self.bt2t['velocity_backward'], 'specification': velocity, 'color': np.array([0, 1, 0])}
                for velocity in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # goal
        if 'goal_forward' in self.task_variants:
            goals = np.linspace(self.pos_x[0], self.pos_x[1], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_forward'], 'specification': goal, 'color': np.array([1, 1, 0])}
                          for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'goal_backward' in self.task_variants:
            goals = np.linspace(-self.pos_x[1], -self.pos_x[0], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_backward'], 'specification': goal, 'color': np.array([0, 1, 1])}
                          for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # flipping
        if 'flip_forward' in self.task_variants:
            goals = np.linspace(self.velocity_y[0], self.velocity_y[1], num=sum(num_tasks_per_subtask))
            tasks_flipping = [
                {'base_task': self.bt2t['flip_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0])} for
                goal in goals]
            np.random.shuffle(tasks_flipping)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_flipping[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # stand_up
        if 'stand_front' in self.task_variants:
            goals = np.linspace(self.pos_y[0], self.pos_y[1], num=sum(num_tasks_per_subtask))
            tasks_stand_up = [
                {'base_task': self.bt2t['stand_front'], 'specification': goal, 'color': np.array([1., 0, 0.5])} for goal
                in goals]
            np.random.shuffle(tasks_stand_up)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_stand_up[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        if 'stand_back' in self.task_variants:
            goals = np.linspace(-self.pos_y[1], -self.pos_y[0], num=sum(num_tasks_per_subtask))
            tasks_stand_up = [
                {'base_task': self.bt2t['stand_back'], 'specification': goal, 'color': np.array([0.5, 0, 1.])} for goal
                in goals]
            np.random.shuffle(tasks_stand_up)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_stand_up[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # jump
        if 'jump' in self.task_variants:
            goals = np.linspace(self.velocity_z[0], self.velocity_z[1], num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['jump'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])}
                          for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # direction
        if 'direction_forward' in self.task_variants:
            goals = np.array([1.] * sum(num_tasks_per_subtask))
            tasks_jump = [
                {'base_task': self.bt2t['direction_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.])}
                for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]
        if 'direction_backward' in self.task_variants:
            goals = np.array([-1.] * sum(num_tasks_per_subtask))
            tasks_jump = [
                {'base_task': self.bt2t['direction_backward'], 'specification': goal, 'color': np.array([0.5, 0., 0.5])}
                for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]
        if 'velocity' in self.task_variants:
            goals = np.linspace(0.0, 3.0, num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['velocity'], 'specification': goal, 'color': np.array([0.5, 0., 0.5])}
                          for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[
                                                             num_tasks_per_subtask_cumsum[i - 1] if i - 1 >= 0 else 0:
                                                             num_tasks_per_subtask_cumsum[i]]

        # Return nested list only if list is given as input
        return tasks if len(num_tasks_list) > 1 else tasks[0]

    def set_meta_mode(self, mode):
        self.meta_mode = mode
