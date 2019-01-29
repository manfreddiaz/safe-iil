import ast
import pickle

from imitation.training._settings import *
from imitation.training._summary import *
from imitation.training._parametrization import PARAMETRIZATIONS_NAMES
from imitation.training._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES


def summarize_iteration(disk_entry, label):
    reading = True
    data_file = open(disk_entry, mode='rb')

    iteration_summary = IterationSummary(label=label)
    while reading:
        try:
            episode_data = pickle.load(data_file)
            summary = EpisodeSummary(label=None)
            for data in episode_data:
                summary.process(data)
            iteration_summary.add_episode_summary(summary)
        except EOFError:
            reading = False

    return iteration_summary


def algorithm_and_parametrization_by_optimization(config):

    algorithm = ALGORITHMS[config.algorithm]
    parametrization = PARAMETRIZATIONS_NAMES[config.parametrization]

    print('{} - {}'.format(algorithm, parametrization))

    summaries = []
    #for optimization_method in range(len(OPTIMIZATION_METHODS_NAMES)):
    disk_entry = experimental_entry(
        algorithm=algorithm,
        experiment_iteration=config.iteration,
        parametrization_name=parametrization,
        horizon=HORIZONS[config.horizon],
        episodes=EPISODES[config.horizon],
        optimization_name=OPTIMIZATION_METHODS_NAMES[config.optimization],
        learning_rate=LEARNING_RATES[config.learning_rate],
        metadata=ast.literal_eval(config.metadata)
    )

    if config.training:
        file = 'training.log'
    else:
        file = 'testing.log'

    summaries.append(summarize_iteration(disk_entry + file, label=OPTIMIZATION_METHODS_NAMES[config.optimization]))

    return summaries

def to_str(l):
    return ', '.join(str(i) for i in l)

def stats_summaries(summaries):
    for iteration_summary in summaries:
        print('method: {}'.format(iteration_summary.label))
        print('\treward: {}'.format(iteration_summary.reward()))
        # print('\t\t  per episode:')
        # print('     {}'.format(iteration_summary.reward_history()))
        print('  queries: {}'.format(iteration_summary.queries()))
        print('\t\t per episode: \r\n[{}]'.format(to_str(iteration_summary.queries_history())))
        print('  penalties: {}'.format(iteration_summary.penalties()))
        print('\t\t per episode: \r\n[{}]'.format(to_str(iteration_summary.penalties_history())))
        # print('  out bounds: {}'.format(iteration_summary.out_bounds()))
        # print('\t\t per episode: \r\n{}'.format(iteration_summary.out_bounds_history()))
        # print('  delta v_l: {}'.format(iteration_summary.delta_v_l()))
        # print('\t \t per episode: \r\n{}'.format(iteration_summary.delta_v_r_history()))
        # print('  delta v_r: {}'.format(iteration_summary.delta_v_r()))
        # print('\t \t per episode: \r\n{}'.format(iteration_summary.delta_v_l_history()))
        print('  no_control: {}'.format(iteration_summary.no_control()))
        print('\t \t per episode: \r\n[{}]'.format(to_str(iteration_summary.no_control_history())))


def render_summaries(summaries):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2)
    figure.canvas.set_window_title('performance')

    axes[0][0].set_title('reward')
    axes[0][1].set_title('queries')
    axes[1][0].set_title('penalties')
    axes[1][1].set_title('out_bounds')

    axes[2][0].set_title('v_diff')
    axes[2][1].set_title('theta_diff')

    for iteration_summary in summaries:
        episode_range = np.arange(0, iteration_summary.episodes())
        axes[0][0].plot(episode_range, iteration_summary.reward_history(), label=iteration_summary.label)
        axes[0][1].plot(episode_range, iteration_summary.queries_history(), label=iteration_summary.label)
        axes[1][0].plot(episode_range, iteration_summary.penalties_history(), label=iteration_summary.label)
        axes[1][1].plot(episode_range, iteration_summary.out_bounds_history(), label=iteration_summary.label)

        axes[2][0].plot(episode_range, iteration_summary.delta_v_l_history(), label=iteration_summary.label)
        axes[2][1].plot(episode_range, iteration_summary.delta_v_r_history(), label=iteration_summary.label)

    for axis in axes:
        axis[0].legend()
        axis[1].legend()

    plt.show()


if __name__ == '__main__':
    parser = process_args()
    parser.add_argument('--training', '-t', action='store_true')
    config = parser.parse_args()

    summaries = algorithm_and_parametrization_by_optimization(config)

    stats_summaries(summaries)