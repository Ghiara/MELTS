import metaworld
import random
import mujoco_py
import numpy as np
from meta_rand_envs.base import MetaEnvironment
import meta_rand_envs.metaworld_benchmarks as mw_bench

# based on repo master from https://github.com/rlworkgroup/metaworld on commit: 2020/10/29 @ 11:17PM, title " Update pick-place-v2 scripted-policy success (#251)", id: 5bcc76e1d455b8de34a044475c9ea3979ca53e2d

class MetaWorldEnv(MetaEnvironment):
    def __init__(self, *args, **kwargs):
        self.metaworld_env = None
        ml10or45 = kwargs['ml10or45']
        self.scripted = kwargs['scripted_policy']
        if ml10or45 == 10:
            if self.scripted:
                self.ml_env = mw_bench.ML10()
            else:
                self.ml_env = metaworld.ML10()
            num_train_tasks_per_base_task = round(kwargs['n_train_tasks'] / 10 + 0.5)
            num_test_tasks_per_base_task = round(kwargs['n_eval_tasks'] / 5 + 0.5)
        elif ml10or45 == 45:
            self.ml_env = metaworld.ML45()
            num_train_tasks_per_base_task = round(kwargs['n_train_tasks'] / 45 + 0.5)
            num_test_tasks_per_base_task = round(kwargs['n_eval_tasks'] / 5 + 0.5)
        elif ml10or45 == 1:
            self.ml_env = metaworld.ML1(kwargs['base_task'])
            num_train_tasks_per_base_task = round(kwargs['n_train_tasks'])
            num_test_tasks_per_base_task = round(kwargs['n_eval_tasks'])
        elif ml10or45 == 3:
            self.ml_env = mw_bench.ML3()
            num_train_tasks_per_base_task = round(kwargs['n_train_tasks'] / 3 + 0.5)
            num_test_tasks_per_base_task = round(kwargs['n_eval_tasks'])
        elif ml10or45 == 2:
            self.ml_env = mw_bench.ML2()
            num_train_tasks_per_base_task = round(kwargs['n_train_tasks'] / 2 + 0.5)
            num_test_tasks_per_base_task = round(kwargs['n_eval_tasks'] / 2 + 0.5)
        else:
            raise NotImplementedError

        self.sample_tasks(num_train_tasks_per_base_task, num_test_tasks_per_base_task)

        self.name2number = {}
        counter = 0
        for t in self.tasks:
            if t.env_name not in self.name2number:
                self.name2number[t.env_name] = counter
                counter += 1

    def sample_tasks(self, num_train_tasks_per_base_task, num_test_tasks_per_base_task):
        self.train_tasks = []
        for name, env_cls in self.ml_env.train_classes.items():
            # print("%d train tasks for  %s" % (len([task for task in self.ml_env.train_tasks if task.env_name == name]), name))
            tasks = random.sample([task for task in self.ml_env.train_tasks if task.env_name == name], num_train_tasks_per_base_task)
            self.train_tasks += tasks

        self.test_tasks = []
        for name, env_cls in self.ml_env.test_classes.items():
            # print("%d train tasks for  %s" % (len([task for task in self.ml_env.test_tasks if task.env_name == name]), name))
            tasks = random.sample([task for task in self.ml_env.test_tasks if task.env_name == name], num_test_tasks_per_base_task)
            self.test_tasks += tasks

        self.tasks = self.train_tasks + self.test_tasks
        if self.scripted:
            self.train_tasks = self.train_tasks + self.test_tasks
        self.reset_task(0)

    def reset_task(self, idx):
        # close window to avoid mulitple windows open at once
        if hasattr(self, 'viewer'):
            self.close()

        task = self.tasks[idx]
        if task.env_name in self.ml_env.train_classes:
            self.metaworld_env = self.ml_env.train_classes[task.env_name]()
        elif task.env_name in self.ml_env.test_classes:
            self.metaworld_env = self.ml_env.test_classes[task.env_name]()

        self.metaworld_env.viewer_setup = self.viewer_setup
        self.metaworld_env.set_task(task)
        self.metaworld_env.reset()
        self.active_env_name = task.env_name
        self.reset()

    def step(self, action):
        ob, reward, done, info = self.metaworld_env.step(action)
        info['true_task'] = dict(base_task=self.name2number[self.active_env_name], specification=self.metaworld_env._target_pos.sum(), target=self.metaworld_env._target_pos, name=self.active_env_name)
        return ob.astype(np.float32), reward, done, info

    def reset(self):
        unformated = self.metaworld_env.reset()
        return unformated.astype(np.float32)

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
        self.viewer.cam.type = 0
        self.metaworld_env.viewer.cam.azimuth = -20
        self.metaworld_env.viewer.cam.elevation = -20

    def __getattr__(self, attrname):
        return getattr(self.metaworld_env, attrname)