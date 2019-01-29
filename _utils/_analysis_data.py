import pickle


from imitation.training._settings import *
from imitation.training._summary import *
from imitation.training._parametrization import PARAMETRIZATIONS_NAMES
from imitation.training._optimization import OPTIMIZATION_METHODS_NAMES, LEARNING_RATES

def summarize_dataset(disk_entry):
    reading = True
    data_file = open(disk_entry, mode='rb')

    v_histograms = []
    theta_histograms = []

    while reading:
        try:
            _, actions, loss = pickle.load(data_file)
            actions = np.array(actions)
            v_histograms.append(np.histogram(actions[::, 0], bins=50, range=[-1, 1]))
            theta_histograms.append(np.histogram(actions[::, 1], bins=50))
        except EOFError:
            reading = False

    return v_histograms, theta_histograms


def plot_histograms(histograms):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for index, histogram in enumerate(histograms):
        values, bins = histogram
        ax.bar(bins[:-1], values, zs=np.repeat(index, 50), zdir='y', width=0.05, alpha=0.5, label=str(index))

    plt.legend()
    plt.show()

if __name__ == '__main__':
    iteration = 0
    horizon_iteration = 0
    parametrization_iteration = 0
    optimization_iteration = 2
    learning_rate_iteration = 0

    algorithm = 'supervised'

    disk_entry = experimental_entry(
        algorithm=algorithm,
        experiment_iteration=iteration,
        parametrization_name=PARAMETRIZATIONS_NAMES[parametrization_iteration],
        horizon=HORIZONS[horizon_iteration],
        episodes=EPISODES[horizon_iteration],
        optimization_name=OPTIMIZATION_METHODS_NAMES[optimization_iteration],
        learning_rate=LEARNING_RATES[learning_rate_iteration]
    )

    v_histograms, theta_histogram = summarize_dataset(disk_entry + 'dataset_evolution.pkl')

    plot_histograms(theta_histogram)