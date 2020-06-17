import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_points(data, title, color='forestgreen'):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(data[:, 0], data[:, 1], color=color, s=3)
    ax.set_title(title)

    restricted_area = patches.Rectangle((130, 70), 20, -20,
                                        linewidth=1,
                                        alpha=0.5,
                                        edgecolor='sandybrown',
                                        facecolor='sandybrown')
    ax.add_patch(restricted_area)
    ax.set(xlim=(np.min(data[:, 0]), np.max(data[:, 0])),
           ylim=(np.min(data[:, 1]), np.max(data[:, 1])))
    plt.show()


def main():
    data_train = np.load('../data/FireEvac_train_set.npy')
    data_test = np.load('../data/FireEvac_test_set.npy')
    reconst_x = np.load('reconst_x.npy')

    # Plot data set
    plot_points(data_train, 'Points on train set')
    plot_points(data_test, 'Points on test set')

    # run VAE model
    # reconstruct test set and plot
    plot_points(reconst_x, 'Reconstructed Points', 'mediumpurple')
    # TODO: generate 1000 samples



if __name__ == '__main__':
    main()
