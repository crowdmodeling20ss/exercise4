import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Util import read_file


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
    ax.scatter(X[:, 0], X[:, 1], label='Point')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc='upper left')

    # Mark the center of data set
    ax.plot(mean_d1, mean_d2, 'o', markersize=10, color='red', alpha=0.5)

    # Draw the direction of two principal components
    V_T = V.T
    plt.arrow(mean_d1, mean_d2, V_T[0, 0], V_T[1, 0], width=0.01, color='red', alpha=0.5)
    plt.arrow(mean_d1, mean_d2, V_T[0, 1], V_T[1, 1], width=0.01, color='blue', alpha=0.5);

    # Show eigenvalues of Sigma
    plt.text(V_T[0, 0] - 1.5, V_T[1, 0] + 0.1, str(sigma[0]), fontsize=12, color='red')
    plt.text(V_T[0, 1], V_T[1, 1] - 0.1, str(sigma[1]), fontsize=12, color='blue')
    plt.show()


def part_2():
    # Image operation in python: https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
    # TODO: Show RGB image via pyplot
    # TODO: Show grayscale image via pyplot
    # TODO: investigate: Is the PIL better than misc? misc.imresize is deprecated.
    image = Image.open('data/PIXNIO-28860-1536x1152.jpeg') \
        .convert('L') \
        .resize((249, 185))
    image.show()
    data = np.asarray(image)

    # TODO: run SVD
    # TODO: Make Reconstruction  (a) all principal components.
    # TODO: Make Reconstruction  (b) 120 principal components.
    # TODO: Make Reconstruction  (c) 50 principal components.
    # TODO: Make Reconstruction  (d) 10 principal components.
    # TODO: At what number is the “energy” lost through truncation smaller than 1%?


def part_3():
    # Read data
    X = read_file('data_DMAP_PCA_vadere.txt')

    # TODO: Visualize the path of the first two pedestrians in the two-dimensional space. What do you observe?

    # Make PCA analysis via SVD
    U, sigma, V = np.linalg.svd(X, 0)
    S = np.diag(sigma)
    trace = S.trace()

    for s in sigma:
        print("Energy:" + str(s / trace))


def main():
    part_1()
    part_2()
    part_3()


if __name__ == '__main__':
    main()
