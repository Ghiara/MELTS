import numpy as np
from meta_rand_envs.base import RandomEnv
from gym import utils
import mujoco_py, os


class AntMultiEnv(RandomEnv, utils.EzPickle):
    def __init__(self, *args, **kwargs):
        self.meta_mode = 'train'
        self.change_mode = kwargs.get('change_mode', '')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 80)
        self.termination_possible = kwargs.get('termination_possible', False)

        self.steps = 0

        # velocity/position specifications in ranges [from, to]
        self.velocity_x = [1.0, 3.0]
        self.pos_x = [5.0, 15.0]
        self.velocity_y = [1. * np.pi, 3. * np.pi]
        self.pos_y = [np.pi / 6., np.pi / 2.]
        self.velocity_z = [0.5, 2.]

        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.base_task = 0
        self.task_specification = 1.0
        task_names = ['velocity_left',
                      'velocity_right',
                      'velocity_up',
                      'velocity_down',
                      'goal_left',
                      'goal_right',
                      'goal_up',
                      'goal_down',
                      'jump',
                      'goal_2D', 'direction_forward', 'direction_backward']
        self.task_variants = kwargs.get('task_variants', task_names)
        self.bt2t = {k : self.task_variants.index(k) if k in self.task_variants else -1 for k in task_names}

        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

        self.model_path = os.path.join(os.getcwd(), 'submodules', 'meta_rand_envs', 'meta_rand_envs', 'ant.xml')
        RandomEnv.__init__(self, kwargs.get('log_scale_limit', 0), self.model_path, 5, hfield_mode=kwargs.get('hfield_mode', 'gentle'), rand_params=[])
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

        torso_xyz_before = np.array(self.get_body_com("torso"))
        try:
            self.do_simulation(action, self.frame_skip)
        except:
            raise RuntimeError("Simulation error, common error is action = nan")

        posafter = self.sim.data.qpos
        velafter = self.sim.data.qvel

        if self.base_task in [self.bt2t['velocity_left'], self.bt2t['velocity_right'], self.bt2t['velocity_up'], self.bt2t['velocity_down']]: #'velocity'
            reward_run = - np.square(velafter[0:2] - self.task_specification).sum()
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.square(self.task_specification).sum()

        elif self.base_task in [self.bt2t['goal_left'], self.bt2t['goal_right'], self.bt2t['goal_up'], self.bt2t['goal_down']]: # 'goal'
            reward_run = - np.square(posafter[0:2] - self.task_specification).sum()
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.square(self.task_specification).sum()

        elif self.base_task == self.bt2t['jump']:  # 'jump'
            reward_run = - np.abs(np.abs(velafter[2]) - self.task_specification)
            reward_ctrl = -0.5 * 1e-1 * np.sum(np.square(action))
            reward = reward_ctrl * 1.0 + reward_run / np.abs(self.task_specification)

        # goal and direction
        elif self.base_task == self.bt2t['goal_2D']:  # 'goal_2D'
            xposafter = np.array(self.get_body_com("torso"))
            reward_run = -np.sum(np.abs(xposafter[:2] - self.task_specification))  # make it happy, not suicidal

            reward_ctrl = - .1 * np.square(action).sum()
            contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 0.0
            reward = reward_run + reward_ctrl - contact_cost + survive_reward

        elif self.base_task in [self.bt2t['direction_forward'], self.bt2t['direction_backward']]:  # 'goal_2D'

            direct = (np.cos(self.task_specification), np.sin(self.task_specification))

            torso_xyz_after = np.array(self.get_body_com("torso"))
            torso_velocity = torso_xyz_after - torso_xyz_before
            reward_run = np.dot((torso_velocity[:2] / self.dt), direct)

            reward_ctrl = - .5 * np.square(action).sum()
            contact_cost = 0.5 * 1e-3 * np.sum(
                np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
            survive_reward = 1.0
            reward = reward_run + reward_ctrl - contact_cost + survive_reward

        else:
            raise RuntimeError("bask task not recognized")

        ob = self._get_obs()

        # print(str(self.base_task) + ": " + str(reward))
        # compared to gym original, we have the possibility to terminate, if the cheetah lies on the back
        if self.termination_possible:
            state = self.state_vector()
            notdone = np.isfinite(state).all() and state[2] >= 0.30 and state[2] <= 10.0 # original gym values: 0.2 and 1.0
            done = not notdone
        else:
            done = False
        self.steps += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, true_task=dict(base_task=self.base_task, specification=self.task_specification))


    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ]).astype(np.float32).flatten() #.astype(np.float32).flatten() added to make compatible

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
        if 'velocity_left' in self.task_variants:
            velocities = np.linspace(-self.velocity_x[1], -self.velocity_x[0], num=sum(num_tasks_per_subtask))
            tasks_velocity = [{'base_task': self.bt2t['velocity_left'], 'specification': np.array([vel, 0]), 'color': np.array([1, 0, 0])} for vel in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'velocity_right' in self.task_variants:
            velocities = np.linspace(self.velocity_x[0], self.velocity_x[1], num=sum(num_tasks_per_subtask))
            tasks_velocity = [{'base_task': self.bt2t['velocity_right'], 'specification': np.array([vel, 0]), 'color': np.array([1, 1, 0])} for vel in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'velocity_up' in self.task_variants:
            velocities = np.linspace(self.velocity_x[0], self.velocity_x[1], num=sum(num_tasks_per_subtask))
            tasks_velocity = [{'base_task': self.bt2t['velocity_up'], 'specification': np.array([0, vel]), 'color': np.array([1, 0, 1])} for vel in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'velocity_down' in self.task_variants:
            velocities = np.linspace(-self.velocity_x[1], -self.velocity_x[0], num=sum(num_tasks_per_subtask))
            tasks_velocity = [{'base_task': self.bt2t['velocity_down'], 'specification': np.array([0, vel]), 'color': np.array([1, 0.5, 0.5])} for vel in velocities]
            np.random.shuffle(tasks_velocity)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_velocity[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]

        # goal
        if 'goal_left' in self.task_variants:
            goals = np.linspace(-self.pos_x[1], -self.pos_x[0], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_left'], 'specification': np.array([goal, 0]), 'color': np.array([1, 0.5, 1])} for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'goal_right' in self.task_variants:
            goals = np.linspace(self.pos_x[0], self.pos_x[1], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_right'], 'specification': np.array([goal, 0]), 'color': np.array([0.5, 1, 1])} for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'goal_up' in self.task_variants:
            goals = np.linspace(self.pos_x[0], self.pos_x[1], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_up'], 'specification': np.array([0, goal]), 'color': np.array([1, 1, 0.5])} for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'goal_down' in self.task_variants:
            goals = np.linspace(-self.pos_x[1], -self.pos_x[0], num=sum(num_tasks_per_subtask))
            tasks_goal = [{'base_task': self.bt2t['goal_down'], 'specification': np.array([0, goal]), 'color': np.array([0.5, 0.5, 0.5])} for goal in goals]
            np.random.shuffle(tasks_goal)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_goal[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]

        # jump
        if 'jump' in self.task_variants:
            goals = np.linspace(self.velocity_z[0], self.velocity_z[1], num=sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['jump'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])} for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]

        # goal_2D
        if 'goal_2D' in self.task_variants:

            a = np.random.random(sum(num_tasks_per_subtask)) * 2 * np.pi
            r = 3 * np.random.random(sum(num_tasks_per_subtask)) ** 0.5
            goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
            tasks_jump = [{'base_task': self.bt2t['goal_2D'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])} for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]

        # direction/vel
        if 'direction_forward' in self.task_variants:
            goals = np.array([0.] * sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['direction_forward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])} for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]
        if 'direction_backward' in self.task_variants:
            goals = np.array([np.pi] * sum(num_tasks_per_subtask))
            tasks_jump = [{'base_task': self.bt2t['direction_backward'], 'specification': goal, 'color': np.array([0.5, 0.5, 0.5])} for goal in goals]
            np.random.shuffle(tasks_jump)
            for i in range(len(num_tasks_list)): tasks[i] += tasks_jump[num_tasks_per_subtask_cumsum[i-1] if i - 1 >= 0 else 0:num_tasks_per_subtask_cumsum[i]]


        # Return nested list only if list is given as input
        return tasks if len(num_tasks_list) > 1 else tasks[0]

    def set_meta_mode(self, mode):
        self.meta_mode = mode
