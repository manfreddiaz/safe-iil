import argparse
import os

import tensorflow as tf

from gym_duckietown.envs import DuckietownEnv
from ._utils._loggers import Logger
from .algorithms import AggreVaTe, DropoutDAgger, DAgger, UPMSDAgger, UPMS, BehavioralCloning
from .learners import NeuralNetworkPolicy, UARandomExploration
from .learners.parametrizations.tf import MonteCarloDropoutResnetOneRegression
from .teachers import UAPurePursuitPolicy

HORIZON = 512
EPISODES = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10
MAP_NAME = 'udem1'
SATURATION_POINT = 4

MIXING_DECAYS = [0.5, 0.65, 0.8, 0.95]
UNCERTAINTY_THRESHOLDS = [0.005, 0.05, 0.5, 1, 5]

# total samples 16384
SUBMISSION_DIRECTORY = 'rss2019'
DEBUG = os.getenv('DEBUG', 0)


def experimental_entry(algorithm, seed):
    # rss2019/dagger/0/
    entry = '{}/{}/{}/'.format(
        SUBMISSION_DIRECTORY,
        algorithm,
        seed
    )

    return entry


def learning_system(algorithm, seed):

    env = DuckietownEnv(
        domain_rand=False,
        max_steps=HORIZON,
        map_name=MAP_NAME,
        seed=seed
    )

    learner = NeuralNetworkPolicy(
        parametrization=MonteCarloDropoutResnetOneRegression(seed=seed),
        optimizer=tf.train.AdamOptimizer(learning_rate=LEARNING_RATE),
        storage_location=experimental_entry(
            algorithm=algorithm,
            seed=seed,
        ),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    teacher = UAPurePursuitPolicy(
        env=env
    )

    return env, learner, teacher


def behavioral(seed):
    env, teacher, learner = learning_system('behavioral', seed)

    return BehavioralCloning(env=env,
                             teacher=teacher,
                             learner=learner,
                             horizon=HORIZON,
                             episodes=EPISODES)


def dagger(mixing_decay, seed):
    env, teacher, learner = learning_system('dagger', seed)

    return DAgger(env=env,
                  teacher=teacher,
                  learner=learner,
                  horizon=HORIZON,
                  episodes=EPISODES,
                  alpha=mixing_decay)


def aggrevate(mixing_decay, seed):
    env, teacher, learner = learning_system('aggrevate', seed)

    return AggreVaTe(env=env,
                     teacher=teacher,
                     learner=learner,
                     explorer=UARandomExploration(seed=seed),
                     horizon=HORIZON,
                     episodes=EPISODES,
                     alpha=mixing_decay,
                     seed=seed)


def dropout_dagger(uncertainty_threshold, seed):
    env, teacher, learner = learning_system('dropout_dagger', seed)

    return DropoutDAgger(env=env,
                         teacher=teacher,
                         learner=learner,
                         horizon=HORIZON,
                         episodes=EPISODES,
                         threshold=uncertainty_threshold)


def upms_dagger(uncertainty_threshold, seed):
    env, teacher, learner = learning_system('upms_dagger', seed)

    return UPMSDAgger(env=env,
                      teacher=teacher,
                      learner=learner,
                      explorer=learner,
                      horizon=HORIZON,
                      episodes=EPISODES,
                      safety_coefficient=SATURATION_POINT / uncertainty_threshold)


def upms(uncertainty_threshold, seed):
    env, teacher, learner = learning_system('upms', seed)

    return UPMS(env=env,
                teacher=teacher,
                learner=learner,
                explorer=UARandomExploration(seed=seed),
                horizon=HORIZON,
                episodes=EPISODES,
                safety_coefficient=SATURATION_POINT / uncertainty_threshold)


def train(algorithm):
    logs = Logger(
        routine=algorithm,
        log_file=algorithm.learner.storage_location + 'training.log',
    )

    algorithm.train(debug=DEBUG)


def _behavioral(args):
    train(algorithm=behavioral(args.seed))


def _default(args):
    print(args)


def _safety(args):
    print(args)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', '-i', default=0, type=int)

    sub_parsers = parser.add_subparsers()

    behavioral = sub_parsers.add_parser('behavioral')
    behavioral.set_defaults(func=_behavioral)

    default = sub_parsers.add_parser('default')
    default.add_argument('algorithm', choices=['aggrevate', 'dagger'])
    default.add_argument('--decay', type=float)
    default.set_defaults(func=_default)

    safety = sub_parsers.add_parser('safety')
    safety.add_argument('algorithm')
    safety.add_argument('--decay', type=float)
    safety.add_argument('--threshold', type=float)

    return parser


if __name__ == '__main__':
    config = process_args().parse_args()
    config.func(config)

    #
    # logs = IILTrainingLogger(
    #     env=environment,
    #     routine=algorithm,
    #     log_file=disk_entry + 'training.log',
    #     data_file=disk_entry + 'dataset_evolution.pkl',
    #     horizon=HORIZONS[config.horizon],
    #     episodes=EPISODES[config.horizon]
    # )
    #
    # algorithm.train(debug=DEBUG)
    #
    # environment.close()
