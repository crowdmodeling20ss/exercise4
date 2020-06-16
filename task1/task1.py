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
    mean = X.mean(axis=0, keepdims=True)
    X_centered = X - mean
    mean_centered_d1, mean_centered_d2 = X_centered.mean(0)

    # Make PCA analysis via SVD
    U, sigma, VT = np.linalg.svd(X_centered, 0)
    V = VT.T
    S = np.diag(sigma)
    trace = S.trace()

    S_one_dimension = np.zeros(S.shape)
    S_one_dimension[0][0] = S[0][0]
    X_one_dimension = U.dot(S_one_dimension).dot(VT)
    MSE_one = (X_centered - X_one_dimension) ** 2
    MSE_one = np.sum(MSE_one)
    print("MSE One Dimension: {:.4f}".format(MSE_one ** 2))

    # Approximates one-dimensional linear subspace
    X_1D = U.dot(S).dot(VT[0])
    print("X 1D: " + str(X_1D))
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('1-Dimensional Projection')
    ax.scatter(X_1D, np.zeros(X_1D.shape), label='Projected Data', c="red", s=3)
    plt.xlabel("z")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    print("X_head: " + str(X_one_dimension))
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
    ax.scatter(X[:, 0], X[:, 1], label='Data', c="mediumseagreen", s=3)
    ax.scatter(X_centered[:, 0], X_centered[:, 1], label='Centered Data', c="lightskyblue", s=3)
    ax.scatter(X_one_dimension[:, 0], X_one_dimension[:, 1], label='Projected Data', c="red", s=1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend(loc='upper left')

    # Mark the center of data set
    ax.plot(mean_d1, mean_d2, 'o', markersize=5, color='olivedrab', label='Center of data')
    ax.plot(mean_centered_d1, mean_centered_d2, 'o', markersize=5, color='darkblue', label='Center of centralized data')

    # Draw the direction of two principal components
    plt.arrow(mean_d1, mean_d2, V[0, 0], V[1, 0], width=0.01, color='darkred', alpha=0.5)
    plt.arrow(mean_d1, mean_d2, V[0, 1], V[1, 1], width=0.01, color='darkblue', alpha=0.5);

    # Show eigenvalues of Sigma
    plt.text(V[0, 0] - 0.6, V[1, 0] + 0.1, "{:.4f}".format(sigma[0]), fontsize=12, color='darkred')
    plt.text(V[0, 1], V[1, 1] - 0.1, "{:.4f}".format(sigma[1]), fontsize=12, color='darkblue')
    plt.show()


def part_2():
    # Image operation in python: https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
    # TODO: investigate: Is the PIL better than misc? misc.imresize is deprecated.
    image = Image.open(MAIN_PATH + 'data/PIXNIO-28860-1536x1152.jpeg') \
        .convert('L') \
        .resize((249, 185))
    image.show()
    data = np.asarray(image)

    plt.imshow(data, cmap='gray', vmin=0, vmax=255)

    # Extract the mean
    mean = data.mean(axis=0, keepdims=True)
    data = data - mean
    plt.imshow(data, cmap='gray', vmin=0, vmax=255)

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

    # Reconstruction with 30 principal components.
    S_30 = np.zeros(S.shape)
    for i in range(30):
        S_30[i][i] = sigma[i]
    reconstructed_30 = np.dot(U, np.dot(S_30, V))

    # Reconstruction with 10 principal components.
    S_10 = np.zeros(S.shape)
    for i in range(10):
        S_10[i][i] = sigma[i]
    reconstructed_10 = np.dot(U, np.dot(S_10, V))

    # Image.fromarray(reconstructed_all).show()
    # Image.fromarray(reconstructed_120).show()
    # Image.fromarray(reconstructed_50).show()
    # Image.fromarray(reconstructed_10).show()

    plt.imshow(reconstructed_all, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_all + mean, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_120, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_120 + mean, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_50, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_50 + mean, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_30, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_30 + mean, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_10, cmap='gray', vmin=0, vmax=255)
    plt.imshow(reconstructed_10 + mean, cmap='gray', vmin=0, vmax=255)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Reconstructed images')
    ax1.imshow(reconstructed_all, cmap='gray', vmin=0, vmax=255)
    ax1.set_title("Reconstructed with all principal components")
    ax2.imshow(reconstructed_120, cmap='gray', vmin=0, vmax=255)
    ax2.set_title("Reconstructed with 120 principal components")
    ax3.imshow(reconstructed_50, cmap='gray', vmin=0, vmax=255)
    ax3.set_title("Reconstructed with 50 principal components")
    ax4.imshow(reconstructed_10, cmap='gray', vmin=0, vmax=255)
    ax4.set_title("Reconstructed with 10 principal components")

    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()

    # At what number is the “energy” lost through truncation smaller than 1%? -- 163
    total_energy = 0
    E = np.zeros(sigma.shape)
    for i, s in enumerate(sigma):
        energy = s / trace
        print("Energy of component number " + str(i) + ": " + str(energy))
        total_energy += energy
        if total_energy > 0.99:
            print("We have 99% of the energy, we do not need components after number " + str(i) + ".")
            # break
        E[i] = total_energy * 100

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Total energy level of L-th singular value')
    # plt.scatter(np.arange(0, len(E)), E, c="mediumseagreen", s=3)
    plt.ylabel("Total energy(%)")
    plt.xlabel("L")
    plt.axvline(x=10, color="lightskyblue", linestyle="--", label='Total energy at 10-th singular value')
    plt.axvline(x=50, color="dodgerblue", linestyle="--", label='Total energy at 50-th singular value')
    plt.axvline(x=120, color="darkblue", linestyle="--", label='Total energy at 120-th singular value')

    plt.axhline(y=98.96, color="firebrick", linestyle=":")
    plt.axvline(x=163, color="firebrick", linestyle="--", label='Total energy at 163-th singular value')
    # 164 = 99.03

    x_ticks = np.append(ax.get_xticks(), 10)
    x_ticks = np.append(x_ticks, 50)
    x_ticks = np.append(x_ticks, 120)
    x_ticks = np.array([1, 10, 50, 120, 163])
    ax.set_xticks(x_ticks)

    y_ticks = np.linspace(0.0, 100.0, num=10)
    y_ticks = np.append(y_ticks, 99)
    y_ticks = np.array([8.33, 37.37, 72.43, 93.85, 98.96])
    ax.set_yticks(y_ticks)

    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.legend(loc='bottom right')
    plt.show()

    plt.scatter(np.arange(1, len(E) + 1), E, c="mediumseagreen", s=3)
    plt.show()


def part_3():
    # Read data
    X = read_file('data_DMAP_PCA_vadere.txt')

    # Visualize the path of the first two pedestrians in the two-dimensional space.
    fig = plt.figure(figsize=(10, 7))
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

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
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
    # part_1()
    #part_2()
    part_3()


if __name__ == '__main__':
    main()
