from gym.core import Env
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
import numpy as np
import os
import glfw
import colorsys


class MetaEnvironment:
    def __init__(self, *args, **kwargs):
        self.train_tasks = None
        self.test_tasks = None
        self.tasks = None
        self.task = None
        self.meta_mode = "train"

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def reset_task(self, idx):
        """
        Resets the environment to the one specified through idx.
        Args:
            idx: task of the meta-learning environment
        """
        raise NotImplementedError

    def set_meta_mode(self, mode):
        self.meta_mode = mode


class NonStationaryMetaEnv(MetaEnvironment):
    def __init__(self, *args, **kwargs):
        self.change_mode = kwargs.get('change_mode', 'location')
        self.change_prob = kwargs.get('change_prob', 1.0)
        self.change_steps = kwargs.get('change_steps', 100)
        self.steps = 0

        self.positive_change_point_basis = kwargs.get('positive_change_point_basis', 10)
        self.negative_change_point_basis = kwargs.get('negative_change_point_basis', -10)
        self.change_point_interval = kwargs.get('change_point_interval', 1)
        self.reset_change_points()

        MetaEnvironment.__init__(self, *args, **kwargs)
        #self.active_task = self.task

    def reset_change_points(self):
        self.positive_change_point = self.positive_change_point_basis + np.random.random() * self.change_point_interval
        self.negative_change_point = self.negative_change_point_basis - np.random.random() * self.change_point_interval

    def check_env_change(self):
        """
        Checks if a condition regarding time or location is fulfilled, leading to changes of the environment.
        Calls change_env_specification() when condition is fulfilled.
        """
        # with some probability change goal direction
        if self.change_mode == "time":
            prob = np.random.uniform(0, 1)
            if prob < self.change_prob and self.steps > self.change_steps and self.steps > 0:
                self.change_active_task(step=self.steps)
                self.steps = 0

        # change direction at some position in the world
        if self.change_mode == "location":
            if self.get_body_com("torso")[0] > self.positive_change_point:
                self.change_active_task(dir=1)
                self.positive_change_point = self.positive_change_point + self.positive_change_point_basis + np.random.random() * self.change_point_interval

            if self.get_body_com("torso")[0] < self.negative_change_point:
                self.change_active_task(dir=-1)
                self.negative_change_point = self.negative_change_point + self.negative_change_point_basis - np.random.random() * self.change_point_interval

    def change_active_task(self, step=100, dir=1):
        """
        Choose a new active task from train or test task,
        depending on the meta_mode and set the corresponding specification.
        Only holds until the end to the episode.
        """
        raise NotImplementedError

    def recolor(self):
        """
        Change colors of agent to visualize env changes when rendering.
        """
        pass


class NonStationaryGoalVelocityEnv(NonStationaryMetaEnv):
    def __init__(self, *args, **kwargs):
        self.task_min_velocity = kwargs.get('task_min_velocity', 0.0)
        self.task_max_velocity = kwargs.get('task_max_velocity', 1.0)
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        self.active_task = 1.0

    def change_active_task(self, *args, **kwargs):
        if self.meta_mode == 'train':
            self.active_task = np.random.choice(self.train_tasks)['velocity']
        elif self.meta_mode == 'test':
            self.active_task = np.random.choice(self.test_tasks)['velocity']
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        hue = (1.0 / 3.0) - (
                    self.active_task / self.task_max_velocity / 3.0)  # maps between red (max) and green (min) in hsv color space
        rgb_value_tuple = colorsys.hsv_to_rgb(hue, 1, 1)
        geom_rgba[1:, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        velocities = np.random.uniform(self.task_min_velocity, self.task_max_velocity, size=(num_tasks,))
        # for tests
        #velocities = np.linspace(self.task_min_velocity, self.task_max_velocity, num_tasks)
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks


class NonStationaryGoalDirectionEnv(NonStationaryMetaEnv):
    def __init__(self, *args, **kwargs):
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        self.active_task = 1.0

    def change_active_task(self, *args, **kwargs):
        change_point = kwargs.get('dir', 1.0)
        if self.active_task == 1.0 and change_point == 1:
            self.active_task = -1.0
        elif self.active_task == -1.0 and change_point == -1:
            self.active_task = 1.0
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        if self.active_task == 1.0:
            rgb_value_tuple = np.array([0, 1, 0])
        elif self.active_task == -1.0:
            rgb_value_tuple = np.array([1, 0, 0])
        geom_rgba[1:, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba

    def sample_tasks(self, num_tasks):
        # standard forward/backward
        if num_tasks == 2:
            directions = np.array([-1, 1])
        # continuous learning, only one direction start
        if num_tasks == 1:
            directions = np.array([1])
        if num_tasks == -1:
            directions = np.array([-1])
        tasks = [{'direction': direction} for direction in directions]
        return tasks


class RandomParamEnv(NonStationaryMetaEnv, MujocoEnv):
    def __init__(self, *args, **kwargs):
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
        file_name = kwargs.get('filename', 'walker2d.xml')
        frame_skip = kwargs.get('frame_skip', 4)
        self.log_scale_limit = kwargs.get('log_scale_limit', 0.0)
        self.rand_params = kwargs.get('rand_params', RAND_PARAMS)
        MujocoEnv.__init__(self, file_name, frame_skip)
        utils.EzPickle.__init__(self)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.save_parameters()

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []
        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts
            new_params = {}
            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)
            param_sets.append(new_params)
        return param_sets

    def change_active_task(self, *args, **kwargs):
        if self.meta_mode == 'train':
            self.active_task = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self.active_task = np.random.choice(self.test_tasks)
        self.set_physical_parameters()
        self.recolor(reset=False)

    def set_physical_parameters(self):
        for param, param_val in self.active_task.items():
            param_variable = getattr(self.sim.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            # body_mass
            if param == 'body_mass':
                self.sim.model.body_mass[:] = param_val
            # body_inertia
            if param == 'body_inertia':
                self.sim.model.body_inertia[:] = param_val
            # damping -> different multiplier for different dofs/joints
            if param == 'dof_damping':
                self.sim.model.dof_damping[:] = param_val
            # friction at the body components
            if param == 'geom_friction':
                self.sim.model.geom_friction[:] = param_val
        self.recolor()

    def recolor(self, reset=True):
        geom_rgba = self._init_geom_rgba.copy()
        if reset:
            self.model.geom_rgba[:] = geom_rgba
        else:
            rgb_value_tuple = np.random.random(3)
            geom_rgba[1:, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task
        self.set_physical_parameters()
        self.reset_change_points()
        self.recolor(reset=True)
        self.steps = 0
        self.reset()


class RandomMassParamEnv(NonStationaryMetaEnv, MujocoEnv):
    def __init__(self, *args, **kwargs):
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        self.file_name = kwargs.get('filename', 'walker2d.xml')
        frame_skip = kwargs.get('frame_skip', 4)
        self.log_scale_limit = kwargs.get('log_scale_limit', 0.0)
        MujocoEnv.__init__(self, self.file_name, frame_skip)
        utils.EzPickle.__init__(self)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.save_parameters()

    def save_parameters(self):
        self.init_params = {}
        self.init_params['body_mass'] = self.model.body_mass
        self.init_params['body_inertia'] = self.model.body_inertia


    def sample_tasks(self, n_tasks):
        num_per_task = int(n_tasks / (self.model.body_mass.shape[0] - 1))

        tasks = []
        # body mass
        for element in range(1, self.model.body_mass.shape[0], 1):
            for _ in range(num_per_task):
                task = {}
                task['body_mass'] = self.init_params['body_mass'].copy()
                task['body_inertia'] = self.init_params['body_inertia'].copy()
                multiplier = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit)
                task['body_mass'][element] *= multiplier
                task['body_inertia'][element]  *= multiplier
                task['base_task'] = element
                task['specification'] = multiplier
                tasks.append(task)
        return tasks

    def change_active_task(self, *args, **kwargs):
        if self.meta_mode == 'train':
            self.active_task = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self.active_task = np.random.choice(self.test_tasks)
        self.set_physical_parameters()
        self.recolor(reset=False)

    def set_physical_parameters(self):
        self.sim.model.body_mass[:] = self.active_task['body_mass']
        self.sim.model.body_inertia[:] = self.active_task['body_inertia']
        self.recolor()

    def recolor(self, reset=True):
        a = self.active_task['base_task']
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value_tuple = [1.0,0,0]#np.random.random(3)
        if self.file_name == 'half_cheetah.xml':
            start = self.active_task['base_task'] if self.active_task['base_task'] == 1 else self.active_task['base_task'] + 1
            end = self.active_task['base_task'] + 2
        else:
            start = self.active_task['base_task']
            end = self.active_task['base_task'] + 1
        geom_rgba[start:end, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task
        self.set_physical_parameters()
        self.reset_change_points()
        self.steps = 0
        self.reset()

class RandomDampParamEnv(NonStationaryMetaEnv, MujocoEnv):
    def __init__(self, *args, **kwargs):
        NonStationaryMetaEnv.__init__(self, *args, **kwargs)
        self.file_name = kwargs.get('filename', 'walker2d.xml')
        frame_skip = kwargs.get('frame_skip', 4)
        self.log_scale_limit = kwargs.get('log_scale_limit', 0.0)
        MujocoEnv.__init__(self, self.file_name, frame_skip)
        utils.EzPickle.__init__(self)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.save_parameters()

    def save_parameters(self):
        self.init_params = {}
        self.init_params['dof_damping'] = self.model.dof_damping

    def sample_tasks(self, n_tasks):
        num_per_task = int(n_tasks / (self.model.dof_damping.shape[0] - 3))

        tasks = []
        # body mass
        for element in range(3, self.model.body_mass.shape[0], 1):
            for _ in range(num_per_task):
                task = {}
                task['dof_damping'] = self.init_params['dof_damping'].copy()
                multiplier = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit)
                task['dof_damping'][element] *= multiplier
                task['base_task'] = element
                task['specification'] = multiplier
                tasks.append(task)
        return tasks

    def change_active_task(self, *args, **kwargs):
        if self.meta_mode == 'train':
            self.active_task = np.random.choice(self.train_tasks)
        elif self.meta_mode == 'test':
            self.active_task = np.random.choice(self.test_tasks)
        self.set_physical_parameters()
        self.recolor()

    def set_physical_parameters(self):
        self.sim.model.dof_damping[:] = self.active_task['dof_damping']
        self.recolor()

    def recolor(self):
        geom_rgba = self._init_geom_rgba.copy()
        rgb_value_tuple = [1.0,0,0]#np.random.random(3)
        if self.file_name == 'half_cheetah.xml':
            start = self.active_task['base_task']
            end = self.active_task['base_task'] + 1
        else:
            start = self.active_task['base_task']
            end = self.active_task['base_task'] + 1
        geom_rgba[start:end, :3] = np.asarray(rgb_value_tuple)
        self.model.geom_rgba[:] = geom_rgba

    def reset_task(self, idx):
        self.task = self.tasks[idx]
        self.active_task = self.task
        self.set_physical_parameters()
        self.reset_change_points()
        self.steps = 0
        self.reset()

# --------- not used anymore --------------
class MetaEnv(Env):
    def step(self, *args, **kwargs):
        return self._step(*args, **kwargs)

    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Sets the specified task to the current environment
        Args:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment
        Returns:
            task: task of the meta-learning environment
        """
        raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        """
        Logs env-specific diagnostic information
        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        pass




class RandomEnv(MetaEnv, MujocoEnv):
    """
    This class provides functionality for randomizing the physical parameters of a mujoco model
    The following parameters are changed:
        - body_mass
        - body_inertia
        - damping coeff at the joints
    """
    RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction', 'hfield']
    RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

    def __init__(self, log_scale_limit, file_name, *args, hfield_mode='gentle', rand_params=RAND_PARAMS, **kwargs):

        assert set(rand_params) <= set(self.RAND_PARAMS_EXTENDED), \
            "rand_params must be a subset of " + str(self.RAND_PARAMS_EXTENDED)
        assert hfield_mode in [None, 'hfield', 'hill', 'basin', 'gentle', 'medium', 'flat', 'random', 'random_plateau']
        self.log_scale_limit = log_scale_limit
        self.file_name = file_name
        self.rand_params = rand_params
        self.mode = hfield_mode
        self.spawn_env()

    def spawn_env(self):
        self.initialize = True
        #intermediate_viewer = None
        if hasattr(self, 'viewer'):
            if self.viewer is not None:
                glfw.destroy_window(self.viewer.window)
                self.close()
        try:
            MujocoEnv.__init__(self, self.file_name, 5)
        except:
            full_file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.file_name)
            if not os.path.exists(full_file_name):
                raise IOError("File %s does not exist" % full_file_name)
            MujocoEnv.__init__(self, full_file_name, 5)
        self.initialize = False

    def get_hfield(self):
        # basis hfield array
        field_length = self.sim.model.hfield_size[0, 0].astype(int)
        field_width = self.sim.model.hfield_size[0, 1].astype(int)
        field_half = int(field_length / 2)
        field = np.zeros((field_width, field_length))  # width, lenght as defined by model
        # reference wall for normalization in the back and front
        field[:, 0] = 1
        field[:, -1] = 1

        if self.mode == 'flat':
            x_pos = np.array([])
            y_pos = np.array([])

        elif self.mode == 'random':
            x_pos = np.array(list(range(1, field_length - 1, 1)))
            y_pos = np.random.uniform(0, 0.1, x_pos.shape[0])
            y_pos[field_half - 2:field_half + 2] = 0

        elif self.mode == 'random_plateau':
            x_pos = np.array(list(range(1, field_length - 1, 1)))
            y_pos = np.zeros(x_pos.shape)
            for i in range(int(x_pos.shape[0] / 2)):
                y_pos[2 * i:2 * i + 2] = np.ones(2) * np.random.uniform(0, 0.1)
            y_pos[field_half - 3:field_half + 2] = 0

        elif self.mode == 'basin':
            position = np.random.choice(np.array([0, 10]))
            x_pos = np.array([1, field_half + 2, field_half + 4, field_half + 6, field_half - 1]) + position
            y_pos = np.array([0.05, 0.05, 0, 0.05, 0.05])

        elif self.mode == 'hill':
            height = np.random.uniform(0.1, 0.3)
            position = np.random.choice(np.array([0, 10]))
            x_pos = np.array([5, 10, 15, 20]) + field_half - 3 + position
            y_pos = np.array([0, height, height, 0])

        elif self.mode == 'gentle':
            height = 0.15
            position = np.random.choice(np.array([0, 10]))
            x_pos = np.array([5, 10, 15, 20]) + field_half - 3 + position
            y_pos = np.array([0, height, height, 0])

        elif self.mode == 'medium':
            height = 0.25
            position = np.random.choice(np.array([0, 10]))
            x_pos = np.array([5, 10, 15, 20]) + field_half - 3 + position
            y_pos = np.array([0, height, height, 0])

        elif self.mode == 'hfield':
            x_pos = np.array([1, 3, 6, 9, 12, 20, 23]) + field_half
            y_pos = np.array([0, 0.1, 0.15, 0.15, 0.05, 0.25, 0])

        elif self.mode is None:
            x_pos = np.array([])
            y_pos = np.array([])

        else:
            raise ValueError("Hfield mode invalid")

        for i, x in enumerate(x_pos):
            if i == x_pos.shape[0] - 1:
                continue
            slope = (y_pos[i + 1] - y_pos[i]) / (x_pos[i + 1] - x_pos[i])
            for j in range(x_pos[i], x_pos[i + 1], 1):
                field[:, j] = y_pos[i] + slope * (j - x_pos[i])

        return field.flatten()


    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []

        for _ in range(n_tasks):
            new_params = {}

            # physical parameters of agent
            # body mass -> one multiplier for all body parts
            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.model.body_mass * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = self.model.body_inertia * body_inertia_multiplyers

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.model.dof_damping, dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit, size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.model.geom_friction, dof_damping_multipliers)

            param_sets.append(new_params)

        return param_sets

    def set_task(self, task):
        # parameters of the hfield that can change with every access of the task
        if 'hfield' in self.rand_params:
            # spawn new, otherwise not possible to overwrite hfield
            self.spawn_env()
            self.sim.model.hfield_data[:] = self.get_hfield()

        # per task constant physical parameters
        self.set_physical_parameters(task)

    def set_physical_parameters(self, task):
        for param, param_val in task.items():
            if param == 'direction':
                continue

            param_variable = getattr(self.sim.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'

            # body_mass
            if param == 'body_mass':
                self.sim.model.body_mass[:] = param_val

            # body_inertia
            if param == 'body_inertia':
                self.sim.model.body_inertia[:] = param_val

            # damping -> different multiplier for different dofs/joints
            if param == 'dof_damping':
                self.sim.model.dof_damping[:] = param_val

            # friction at the body components
            if param == 'geom_friction':
                self.sim.model.geom_friction[:] = param_val

        self.cur_params = task

    def get_task(self):
        return self.cur_params

