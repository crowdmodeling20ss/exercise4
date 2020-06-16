import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial
from scipy.linalg import eigh
from sklearn.datasets import make_swiss_roll
from mpl_toolkits.mplot3d import Axes3D
from Util import read_file, MAIN_PATH


import numexpr as ne
import numpy as np
import copy

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
import sklearn.manifold as manifold

# NOTE: make sure "path/to/datafold" is in sys.path or PYTHONPATH if datafold is not installed
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.utils.plot import plot_pairwise_eigenvector
from datafold.dynfold import LocalRegressionSelection

def generate_points(N):
    """
    Generate periodic data set

    :param N:
    :return: [N, 2] numpy array
    """
    # X = {xk ∈ R2}Nk=1, xk = (cos(tk), sin(tk)), tk = (2πk)/(N + 1).
    data = []
    tk_arr = []
    for k in range(1, N + 1):
        t_k = (2 * math.pi * k) / (N + 1)  # tk = (2πk)/(N + 1).
        x_k = (math.cos(t_k), math.sin(t_k))
        data.append(x_k)
        tk_arr.append(t_k)
    X = np.array(data)

    return X, tk_arr

def plot_generated_points(X):
    fig,ax = plt.subplots(1,1, figsize=(12, 12))
    ax.scatter(X[:, 0], X[:, 1], color='mediumpurple', s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig('part_1_generated_points.png')
    plt.show()


def distance_matrix(X):
    """
    Form a distance matrix D with entries

    :param X: [N, 2] numpy array of data points
    :return: [N, N] distance matrix for each entry.
            Important!!: D contains squares of the distances
    """
    D = scipy.spatial.distance.pdist(X, metric='sqeuclidean')
    D = scipy.spatial.distance.squareform(D)

    return D

def diffusion_map_algorithm(X,L):
    """

    :param X: Data points [N,]
    :param L: Number of eigenfunctions
    :return: [N, L+1] matrix. λ0φ0, λ1φ1, ...., λL+1φL+1 for every data point
    """

    # 1. Form a distance matrix D with entries
    D = distance_matrix(X)

    # 2. Set ε to 5% of the diameter of the data set: ε = 0.05(maxi,j Di,j )
    epsilon = np.sqrt(np.max(D)) * 0.05

    # 3. Form the kernel matrix W with W = exp 􏰀−D^2 /ε􏰁.
    W = np.exp(-D / epsilon)

    # 4. Form the diagonal normalization matrix Pii = 􏰄Nj=1 Wij.
    P = np.sum(W, axis=1)

    # 5. Normalize W to form the kernel matrix K = P^−1 W P^−1
    P_inverse = np.linalg.inv(np.diag(P))
    K = np.matmul(np.matmul(P_inverse, W), P_inverse)
    # K = P_inverse.dot(P_inverse).dot(W).dot(P_inverse)

    # 6. Form the diagonal normalization matrix Qii = 􏰄Nj=1 Kij.
    Q = np.sum(K, axis=1)

    # 7. Form the symmetric matrix Tˆ = Q−1/2KQ−1/2.

    Q_inverse = np.sqrt(np.linalg.inv(np.diag(Q)))
    T = np.matmul(np.matmul(Q_inverse, K), Q_inverse)
    # T = Q_inverse.dot(K).dot(Q_inverse)

    # 8. Find the L + 1 largest eigenvalues al and associated eigenvectors vl of T
    # The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
    w, v = eigh(T)
    a_l = w[-(L + 1):]
    v_l = v.T[-(L + 1):]
    # Reverse the vectors to have decreasing order
    a_l = a_l[::-1]
    v_l = v_l[::-1]

    # 9. Compute the eigenvalues of Tˆ1/ε by λ2 = a1/ε.
    lambda_l = np.sqrt(a_l ** (1 / epsilon))

    # 10. Compute the eigenvectors φl of the matrix T = Q−1K by φl = Q−1/2vl.
    S = np.matmul(Q_inverse, v_l.T)

    return S, lambda_l, v_l, a_l, Q, Q_inverse #* lambda_l


def plot_5_eigenfunctions(tk,S):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 12))
    # Skipping φ0 since it is constant
    ax1.plot(tk, S[:, 0], color='teal', linewidth=0.5, label='$\phi_0$')
    ax1.legend(loc="upper right")
    ax1.plot(tk, S[:, 1], color='teal', linewidth=0.5, label='$\phi_1$')
    ax1.legend(loc="upper right")
    ax2.plot(tk, S[:, 2], color='teal', linewidth=0.5, label='$\phi_2$')
    ax2.legend(loc="upper right")
    ax3.plot(tk, S[:, 3], color='teal', linewidth=0.5, label='$\phi_3$')
    ax3.legend(loc="upper right")
    ax4.plot(tk, S[:, 4], color='teal', linewidth=0.5, label='$\phi_4$')
    ax4.legend(loc="upper right")
    ax5.plot(tk, S[:, 5], color='teal', linewidth=0.5, label='$\phi_5$')
    ax5.legend(loc="upper right")
    fig.text(0.5, 0.005, '$t_k$', ha='center')
    fig.text(0.005, 0.5, '$\phi_l(x_k)$', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig('part_1_eigenfunctions_5.png')
    plt.show()

def plot_eigenfunctions(S):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(12,12))
    # Skipping φ0 since it is constant
    ax1.scatter(S[:, 1], S[:, 2], color='mediumpurple', s=1)
    ax1.set_ylabel('$\phi_2$')
    ax2.scatter(S[:, 1], S[:, 3], color='mediumpurple', s=1)
    ax2.set_ylabel('$\phi_3$')
    ax3.scatter(S[:, 1], S[:, 4], color='mediumpurple', s=1)
    ax3.set_ylabel('$\phi_4$')
    ax4.scatter(S[:, 1], S[:, 5], color='mediumpurple', s=1)
    ax4.set_ylabel('$\phi_5$')
    ax5.scatter(S[:, 1], S[:, 6], color='mediumpurple', s=1)
    ax5.set_ylabel('$\phi_6$')
    ax6.scatter(S[:, 1], S[:, 7], color='mediumpurple', s=1)
    ax6.set_ylabel('$\phi_7$')
    ax7.scatter(S[:, 1], S[:, 8], color='mediumpurple', s=1)
    ax7.set_ylabel('$\phi_8$')
    ax8.scatter(S[:, 1], S[:, 9], color='mediumpurple', s=1)
    ax8.set_ylabel('$\phi_9$')
    ax9.scatter(S[:, 1], S[:, 10], color='mediumpurple', s=1)
    ax9.set_ylabel('$\phi_{10}$')

    fig.text(0.5, 0.005, '$\phi_1$', ha='center')
    plt.tight_layout()
    plt.savefig('part_2_eigenfunctions_versus_10_1000p.png')
    plt.show()

def plot_swiss_roll(X,time):
    fig = plt.figure()
    ax0 = fig.gca(projection='3d')
    ax0.scatter(X[:, 0], X[:, 1], X[:, 2], c=time, cmap="twilight", s=1)
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax0.set_title("Swiss Roll")
    plt.show()

def part_1():
    N = 1000
    L = 5
    X, tk = generate_points(N)
    plot_generated_points(X)
    tk = np.array(tk)
    S, _, _, _, _, _ = diffusion_map_algorithm(X, L)
    plot_5_eigenfunctions(tk, S)

def part_2():
    N = 5000
    L = 10
    X, t = make_swiss_roll(N, noise=0.0, random_state=None)
    plot_swiss_roll(X, t)
    S, lambda_l, v_l, a_l, Q, Q_inverse = diffusion_map_algorithm(X, L)
    plot_eigenfunctions(S)

    #U = v_l
    #V = np.matmul(Q_inverse, U)
    R = np.matmul(Q, S)
    plot_swiss_roll(R, t)



    # PCA
    X = X - X.mean(axis=0, keepdims=True)
    U, sigma, V = np.linalg.svd(X, 0)
    S = np.diag(sigma)
    trace = S.trace()

    print("Sigma values of Swiss Roll: ", sigma)

    # Reconstruction with 3 principal components
    energy_3 = 0
    S_3 = np.zeros(S.shape)
    for i in range(3):
        S_3[i][i] = sigma[i]
        energy_3 += sigma[i] / trace
    reconstructed_3 = np.dot(U, np.dot(S_3, V))

    # Reconstruction with 2 principal components
    energy_2 = 0
    S_2 = np.zeros(S.shape)
    for i in range(2):
        S_2[i][i] = sigma[i]
        energy_2 += sigma[i] / trace
    reconstructed_2 = np.dot(U, np.dot(S_2, V))

    fig = plt.figure()
    ax1 = fig.gca(projection='3d')
    ax1.scatter(reconstructed_3[:, 0], reconstructed_3[:, 1], reconstructed_3[:, 2], c=t, cmap="twilight", s=1)
    ax1.set_title("Swiss Roll \n Reconstructed with 3 principal components \n Energy: {:.2f}%".format(energy_3 * 100))
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax2 = fig.gca(projection='3d')
    ax2.scatter(reconstructed_2[:, 0], reconstructed_2[:, 1], reconstructed_2[:, 2], c=t, cmap="twilight", s=1)
    ax2.set_title("Swiss Roll \n Reconstructed with 2 principal components \n Energy: {:.2f}%".format(energy_2 * 100))
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.tight_layout()
    plt.show()

def part_3():
    data = read_file('data_DMAP_PCA_vadere.txt')
    L = 10
    s, _, _, _, _, _ = diffusion_map_algorithm(data, L)
    plot_eigenfunctions(s)

def bonus():
    nr_samples = 15000

    # reduce number of points for plotting
    nr_samples_plot = 1000
    idx_plot = np.random.permutation(nr_samples)[0:nr_samples_plot]

    # generate point cloud
    X, X_color = make_s_curve(nr_samples, random_state=3, noise=0)

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[idx_plot, 0], X[idx_plot, 1], X[idx_plot, 2], c=X_color[idx_plot], cmap=plt.cm.Spectral)
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")
    ax.set_title("point cloud on S-shaped manifold");
    ax.view_init(10, 70)

    X_pcm = pfold.PCManifold(X)
    X_pcm.optimize_parameters(result_scaling=0.5)
    print(f'epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}')

    dmap = dfold.DiffusionMaps(kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon), n_eigenpairs=9,
                               dist_kwargs=dict(cut_off=X_pcm.cut_off))
    dmap = dmap.fit(X_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_

    plot_pairwise_eigenvector(eigenvectors=dmap.eigenvectors_[idx_plot, :], n=1,
                              fig_params=dict(figsize=[15, 15]),
                              scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]))

def main():
    ## Part 1
    #part_1()
    ## Part 2
    #part_2()
    ## Part 3
    #part_3()

    bonus()


if __name__ == '__main__':
    main()
