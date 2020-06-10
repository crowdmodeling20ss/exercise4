import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
sys.path.append('../')
from Util import read_file, MAIN_PATH


def part_1():
    # Read data
    X = read_file('pca_dataset.txt')

    # Find center of data set
    mean_d1, mean_d2 = X.mean(0)

    # Make PCA analysis via SVD
    U, sigma, V = np.linalg.svd(X, 0)
    S = np.diag(sigma)
    trace = S.trace()

    # Approximates one-dimensional linear subspace
    X_head = X.dot(V[:, 0])  # V[:, 0] first column of V

    print("X_head: " + str(X_head))
    print("U: " + str(U))
    print("V: " + str(V))
    print("Sigma: " + str(S))
    print("Trace: " + str(trace))
    print("Energy of " + str(sigma[0]) + ": " + str(sigma[0] / trace))
    print("Energy of " + str(sigma[1]) + ": " + str(sigma[1] / trace))

    # Plot data set
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('PCA')
    ax.scatter(X[:, 0], X[:, 1], label='Points', c="mediumseagreen", s=3)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc='upper left')

    # Mark the center of data set
    ax.plot(mean_d1, mean_d2, 'o', markersize=10, color='red', alpha=0.5)

    # Draw the direction of two principal components
    V_T = V.T
    plt.arrow(mean_d1, mean_d2, V_T[0, 0], V_T[1, 0], width=0.01, color='darkred', alpha=0.5)
    plt.arrow(mean_d1, mean_d2, V_T[0, 1], V_T[1, 1], width=0.01, color='darkblue', alpha=0.5);

    # Show eigenvalues of Sigma
    plt.text(V_T[0, 0] - 0.6, V_T[1, 0] + 0.1, "{:.4f}".format(sigma[0]), fontsize=12, color='darkred')
    plt.text(V_T[0, 1], V_T[1, 1] - 0.1,  "{:.4f}".format(sigma[1]), fontsize=12, color='darkblue')
    plt.show()


def part_2():
    # Image operation in python: https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
    # TODO: investigate: Is the PIL better than misc? misc.imresize is deprecated.
    image = Image.open(MAIN_PATH + '../data/PIXNIO-28860-1536x1152.jpeg') \
        .convert('L') \
        .resize((249, 185))
    image.show()
    data = np.asarray(image)

    # Extract the mean
    data = data - data.mean(axis=0, keepdims=True)

    # Make PCA analysis via SVD
    U, sigma, V = np.linalg.svd(data, 0)
    S = np.diag(sigma)
    trace = S.trace()

    # Reconstruction with all principal components.
    reconstructed_all = np.dot(U, np.dot(S, V))

    # Reconstruction with 120 principal components.
    S_120 = np.zeros(S.shape)
    for i in range(120):
        S_120[i][i] = sigma[i]
    reconstructed_120 = np.dot(U, np.dot(S_120, V))

    # Reconstruction with 50 principal components.
    S_50 = np.zeros(S.shape)
    for i in range(50):
        S_50[i][i] = sigma[i]
    reconstructed_50 = np.dot(U, np.dot(S_50, V))

    # Reconstruction with 10 principal components.
    S_10 = np.zeros(S.shape)
    for i in range(10):
        S_10[i][i] = sigma[i]
    reconstructed_10 = np.dot(U, np.dot(S_10, V))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Reconstructed images')
    ax1.imshow(reconstructed_all)
    ax1.set_title("Reconstructed with all principal components")
    ax2.imshow(reconstructed_120)
    ax2.set_title("Reconstructed with 120 principal components")
    ax3.imshow(reconstructed_50)
    ax3.set_title("Reconstructed with 50 principal components")
    ax4.imshow(reconstructed_10)
    ax4.set_title("Reconstructed with 10 principal components")

    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()

    # At what number is the “energy” lost through truncation smaller than 1%? -- 163
    total_energy = 0
    for i, s in enumerate(sigma):
        energy = s / trace
        print("Energy of component number " + str(i) + ": " + str(energy))
        total_energy += energy
        if total_energy > 0.99:
            print("We have 99% of the energy, we do not need components after number " + str(i) + ".")
            break


def part_3():
    # Read data
    X = read_file('data_DMAP_PCA_vadere.txt')

    # Visualize the path of the first two pedestrians in the two-dimensional space.
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    ax.set_title('Pedestrian Paths')
    ax.plot(X[:, 0], X[:, 1], color='dodgerblue', linewidth=0.5, label='First Pedestrian')
    ax.scatter(X[0, 0], X[0, 1], c='lightskyblue', s=5, label='Starting Point of the First Pedestrian')
    ax.scatter(X[-1, 0], X[-1, 1], c='darkblue', s=5, label='Ending Point of the First Pedestrian')
    ax.plot(X[:, 2], X[:, 3], color='firebrick', linewidth=0.5, label='Second Pedestrian')
    ax.scatter(X[0, 2], X[0, 3], c='lightcoral', s=5, label='Starting Point of the Second Pedestrian')
    ax.scatter(X[-1, 2], X[-1, 3], c='darkred', s=5, label='Ending Point of the Second Pedestrian')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper right')
    plt.show()

    # Make PCA analysis via SVD
    U, sigma, V = np.linalg.svd(X, 0)
    S = np.diag(sigma)
    trace = S.trace()

    # Reconstruction with 2 principal components
    energy_2 = 0
    S_2 = np.zeros(S.shape)
    for i in range(2):
        S_2[i][i] = sigma[i]
        energy_2 += sigma[i] / trace
    reconstructed_2 = np.dot(U, np.dot(S_2, V))

    # Reconstruction with 3 principal components
    energy_3 = 0
    S_3 = np.zeros(S.shape)
    for i in range(3):
        S_3[i][i] = sigma[i]
        energy_3 += sigma[i] / trace
    reconstructed_3 = np.dot(U, np.dot(S_3, V))

    # Reconstruction with 4 principal components
    energy_4 = 0
    S_4 = np.zeros(S.shape)
    for i in range(4):
        S_4[i][i] = sigma[i]
        energy_4 += sigma[i] / trace
    reconstructed_4 = np.dot(U, np.dot(S_4, V))

    # Reconstruction with 5 principal components
    energy_5 = 0
    S_5 = np.zeros(S.shape)
    for i in range(5):
        S_5[i][i] = sigma[i]
        energy_5 += sigma[i] / trace
    reconstructed_5 = np.dot(U, np.dot(S_5, V))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
    ax1.plot(reconstructed_2[:, 0], reconstructed_2[:, 1], color='dodgerblue', linewidth=0.5, label='First Pedestrian')
    ax1.plot(reconstructed_2[:, 2], reconstructed_2[:, 3], color='firebrick', linewidth=0.5, label='Second Pedestrian')
    ax1.set_title("Reconstructed with 2 principal components \n Energy: {:.2f}%".format(energy_2 * 100))
    ax1.legend(loc="upper right")
    ax2.plot(reconstructed_3[:, 0], reconstructed_3[:, 1], color='dodgerblue', linewidth=0.5, label='First Pedestrian')
    ax2.plot(reconstructed_3[:, 2], reconstructed_3[:, 3], color='firebrick', linewidth=0.5, label='Second Pedestrian')
    ax2.set_title("Reconstructed with 3 principal components\n Energy: {:.2f}%".format(energy_3 * 100))
    ax2.legend(loc="upper right")
    ax3.plot(reconstructed_4[:, 0], reconstructed_4[:, 1], color='dodgerblue', linewidth=0.5, label='First Pedestrian')
    ax3.plot(reconstructed_4[:, 2], reconstructed_4[:, 3], color='firebrick', linewidth=0.5, label='Second Pedestrian')
    ax3.set_title("Reconstructed with 4 principal components \n Energy: {:.2f}%".format(energy_4 * 100))
    ax3.legend(loc="upper right")
    ax4.plot(reconstructed_5[:, 0], reconstructed_5[:, 1], color='dodgerblue', linewidth=0.5, label='First Pedestrian')
    ax4.plot(reconstructed_5[:, 2], reconstructed_5[:, 3], color='firebrick', linewidth=0.5, label='Second Pedestrian')
    ax4.set_title("Reconstructed with 5 principal components \n Energy: {:.2f}%".format(energy_5 * 100))
    ax4.legend(loc="upper right")
    fig.text(0.5, 0.01, 'x', ha='center')
    fig.text(0.01, 0.5, 'y', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig('part_3_reconstructed_paths.png')
    plt.show()


def main():
    part_1()
    part_2()
    part_3()


if __name__ == '__main__':
    main()
