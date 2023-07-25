"""Proposal for a simple, understandable MetaWorld API."""
import abc
import pickle
from collections import OrderedDict
from typing import List, NamedTuple, Type

import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
import numpy as np

from metaworld.envs.mujoco.sawyer_xyz.v1 import (
    SawyerNutAssemblyEnv,
    SawyerBasketballEnv,
    SawyerBinPickingEnv,
    SawyerBoxCloseEnv,
    SawyerButtonPressEnv,
    SawyerButtonPressTopdownEnv,
    SawyerButtonPressTopdownWallEnv,
    SawyerButtonPressWallEnv,
    SawyerCoffeeButtonEnv,
    SawyerCoffeePullEnv,
    SawyerCoffeePushEnv,
    SawyerDialTurnEnv,
    SawyerNutDisassembleEnv,
    SawyerDoorEnv,
    SawyerDoorCloseEnv,
    SawyerDoorLockEnv,
    SawyerDoorUnlockEnv,
    SawyerDrawerCloseEnv,
    SawyerDrawerOpenEnv,
    SawyerFaucetCloseEnv,
    SawyerFaucetOpenEnv,
    SawyerHammerEnv,
    SawyerHandInsertEnv,
    SawyerHandlePressEnv,
    SawyerHandlePressSideEnv,
    SawyerHandlePullEnv,
    SawyerHandlePullSideEnv,
    SawyerLeverPullEnv,
    SawyerPegInsertionSideEnv,
    SawyerPegUnplugSideEnv,
    SawyerPickOutOfHoleEnv,
    SawyerPlateSlideEnv,
    SawyerPlateSlideBackEnv,
    SawyerPlateSlideBackSideEnv,
    SawyerPlateSlideSideEnv,
    SawyerPushBackEnv,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerShelfPlaceEnv,
    SawyerSoccerEnv,
    SawyerStickPullEnv,
    SawyerStickPushEnv,
    SawyerSweepEnv,
    SawyerSweepIntoGoalEnv,
    SawyerWindowCloseEnv,
    SawyerWindowOpenEnv,
)
from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerNutAssemblyEnvV2,
    SawyerBasketballEnvV2,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnvV2,
    SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressEnvV2,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2,
    SawyerCoffeePullEnvV2,
    SawyerCoffeePushEnvV2,
    SawyerDialTurnEnvV2,
    SawyerNutDisassembleEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
    SawyerDoorEnvV2,
    SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerFaucetCloseEnvV2,
    SawyerFaucetOpenEnvV2,
    SawyerHammerEnvV2,
    SawyerHandInsertEnvV2,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePressEnvV2,
    SawyerHandlePullSideEnvV2,
    SawyerHandlePullEnvV2,
    SawyerLeverPullEnvV2,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnvV2,
    SawyerPickOutOfHoleEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideSideEnvV2,
    SawyerPlateSlideEnvV2,
    SawyerPushBackEnvV2,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnvV2,
    SawyerSoccerEnvV2,
    SawyerStickPullEnvV2,
    SawyerStickPushEnvV2,
    SawyerSweepEnvV2,
    SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnvV2,
)

EnvName = str


class Task(NamedTuple):
    """All data necessary to describe a single MDP.

    Should be passed into a MetaWorldEnv's set_task method.
    """

    env_name: EnvName
    data: bytes  # Contains env parameters like random_init and *a* goal


class MetaWorldEnv:
    """Environment that requires a task before use.

    Takes no arguments to its constructor, and raises an exception if used
    before `set_task` is called.
    """

    def set_task(self, task: Task) -> None:
        """Set the task.

        Raises:
            ValueError: If task.env_name is different from the current task.

        """


class Benchmark(abc.ABC):
    """A Benchmark.

    When used to evaluate an algorithm, only a single instance should be used.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @property
    def train_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for training."""
        return self._train_classes

    @property
    def test_classes(self) -> 'OrderedDict[EnvName, Type]':
        """Get all of the environment classes used for testing."""
        return self._test_classes

    @property
    def train_tasks(self) -> List[Task]:
        """Get all of the training tasks for this benchmark."""
        return self._train_tasks

    @property
    def test_tasks(self) -> List[Task]:
        """Get all of the test tasks for this benchmark."""
        return self._test_tasks


_ML_OVERRIDE = dict(partially_observable=True)
_MT_OVERRIDE = dict(partially_observable=False)

_N_GOALS = 50


def _encode_task(env_name, data):
    return Task(env_name=env_name, data=pickle.dumps(data))


def _make_tasks(classes, args_kwargs, kwargs_override):
    tasks = []
    for (env_name, args) in args_kwargs.items():
        assert len(args['args']) == 0
        env_cls = classes[env_name]
        env = env_cls()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = args['kwargs'].copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS

        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
    return tasks


# self defined ML2 (reach, push)
ML2_MODE_CLS_DICT = OrderedDict((
    ('train',
     OrderedDict((
         ('reach-v2',  SawyerReachEnvV2),
         ('push-v2', SawyerPushEnvV2)
     ))
     ),
    ('test',
     OrderedDict((
         ('reach-v2',  SawyerReachEnvV2),
         ('push-v2', SawyerPushEnvV2)
     ))
     )
))

ml2_mode_train_args_kwargs = {
    key: dict(args=[], kwargs={
        'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key),
    })
    for key, _ in ML2_MODE_CLS_DICT['train'].items()
}

ml2_mode_test_args_kwargs = {
    key: dict(args=[], kwargs={
        'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)
    })
    for key, _ in ML2_MODE_CLS_DICT['test'].items()
}

ML2_MODE_ARGS_KWARGS = dict(
    train=ml2_mode_train_args_kwargs,
    test=ml2_mode_test_args_kwargs,
)


class ML2(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = ML2_MODE_CLS_DICT['train']
        self._test_classes = ML2_MODE_CLS_DICT['test']

        train_kwargs = ml2_mode_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _ML_OVERRIDE)
        test_kwargs = ml2_mode_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes,
                                       test_kwargs,
                                       _ML_OVERRIDE)


# self defined ML3
ML3_MODE_CLS_DICT = OrderedDict((
    ('train',
     OrderedDict((
         ('reach-v1', SawyerReachPushPickPlaceEnv),
         ('push-v1', SawyerReachPushPickPlaceEnv),
         ('pick-place-v1', SawyerReachPushPickPlaceEnv)))
     ),
    ('test',
     OrderedDict([('sweep-into-v1', SawyerSweepIntoGoalEnv)])
     )
))
ml3_mode_train_args_kwargs = {
    key: dict(args=[], kwargs={
        'task_id': list(ALL_V1_ENVIRONMENTS.keys()).index(key),
    })
    for key, _ in ML3_MODE_CLS_DICT['train'].items()
}

ml3_mode_test_args_kwargs = {
    key: dict(args=[], kwargs={'task_id': list(ALL_V1_ENVIRONMENTS.keys()).index(key)})
    for key, _ in ML3_MODE_CLS_DICT['test'].items()
}

ml3_mode_train_args_kwargs['reach-v1']['kwargs']['task_type'] = 'reach'
ml3_mode_train_args_kwargs['push-v1']['kwargs']['task_type'] = 'push'
ml3_mode_train_args_kwargs['pick-place-v1']['kwargs']['task_type'] = 'pick_place'

ML3_MODE_ARGS_KWARGS = dict(
    train=ml3_mode_train_args_kwargs,
    test=ml3_mode_test_args_kwargs,
)


class ML3(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = ML3_MODE_CLS_DICT['train']
        self._test_classes = ML3_MODE_CLS_DICT['test']

        train_kwargs = ml3_mode_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _ML_OVERRIDE)
        test_kwargs = ml3_mode_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes,
                                       test_kwargs,
                                       _ML_OVERRIDE)


class ML10(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = _env_dict.MEDIUM_MODE_CLS_DICT['train']
        self._test_classes = _env_dict.MEDIUM_MODE_CLS_DICT['test']

        # edit for proper demonstration policies
        self._train_classes['peg-insert-side-v1'] = SawyerPegInsertionSideEnvV2
        self._test_classes['lever-pull-v1'] = SawyerLeverPullEnvV2

        train_kwargs = _env_dict.medium_mode_train_args_kwargs
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _ML_OVERRIDE)
        test_kwargs = _env_dict.medium_mode_test_args_kwargs
        self._test_tasks = _make_tasks(self._test_classes,
                                       test_kwargs,
                                       _ML_OVERRIDE)


class ML45(Benchmark):

    def __init__(self):
        super().__init__()
        self._train_classes = _env_dict.HARD_MODE_CLS_DICT['train']
        self._test_classes = _env_dict.HARD_MODE_CLS_DICT['test']
        train_kwargs = _env_dict.HARD_MODE_ARGS_KWARGS['train']
        self._train_tasks = _make_tasks(self._train_classes,
                                        train_kwargs,
                                        _ML_OVERRIDE)
        self._test_tasks = _make_tasks(self._test_classes,
                                       _env_dict.HARD_MODE_ARGS_KWARGS['test'],
                                       _ML_OVERRIDE)
