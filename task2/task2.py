import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial
from scipy.linalg import eigh
from sklearn.datasets import make_swiss_roll

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

    :param X: Data points
    :param L: Number of eigenfunctions
    :return:
    """

    # 1. Form a distance matrix D with entries
    D = distance_matrix(X)
    # 2. Set ε to 5% of the diameter of the data set: ε = 0.05(maxi,j Di,j )
    epsilon = np.max(D) * 0.05
    # 3. Form the kernel matrix W with W = exp 􏰀−D^2 /ε􏰁.
    W = np.exp(-D / epsilon)
    # 4. Form the diagonal normalization matrix Pii = 􏰄Nj=1 Wij.
    P = np.sum(W, axis=1)
    # 5. Normalize W to form the kernel matrix K = P^−1 W P^−1
    P_inverse = np.linalg.inv(np.diag(P))
    K = P_inverse.dot(W).dot(P_inverse)   #****** There was an additional P_inverse?
    # 6. Form the diagonal normalization matrix Qii = 􏰄Nj=1 Kij.
    Q = np.sum(K, axis=1)
    # 7. Form the symmetric matrix Tˆ = Q−1/2KQ−1/2.
    Q_inverse = np.sqrt(np.linalg.inv(np.diag(Q)))
    T = Q_inverse.dot(K).dot(Q_inverse)
    # 8. Find the L + 1 largest eigenvalues al and associated eigenvectors vl of T
    # The normalized selected eigenvector corresponding to the eigenvalue w[i] is the column v[:,i].
    w, v = eigh(T)
    a_l = w[-(L + 1):]
    v_l = v.T[-(L + 1):]
    # Reverse the vectors to have decreasing order
    a_l = a_l[::-1]
    v_l = v_l[::-1]
    # 9. Compute the eigenvalues of Tˆ1/ε by λ2 = a1/ε.
    lambda_l = np.sqrt(a_l ** (1 / epsilon))  # TODO: Do we need to take square root of a_l?
    print(lambda_l.shape)
    print(lambda_l)
    # 10. Compute the eigenvectors φl of the matrix T = Q−1K by φl = Q−1/2vl.
    S = np.matmul(Q_inverse, v_l.T)
    print(S.shape)
    print(S)
    print((S*lambda_l).shape)
    print(S * lambda_l)
    return S * lambda_l


def plot_5_eigenfunctions(tk,S):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 12))
    # Skipping φ0 since it is constant
    ax1.plot(tk, S[:, 1], color='dodgerblue', linewidth=0.5, label='$φ_1$')
    ax1.legend(loc="upper right")
    ax2.plot(tk, S[:, 2], color='dodgerblue', linewidth=0.5, label='$φ_2$')
    ax2.legend(loc="upper right")
    ax3.plot(tk, S[:, 3], color='dodgerblue', linewidth=0.5, label='$φ_3$')
    ax3.legend(loc="upper right")
    ax4.plot(tk, S[:, 4], color='dodgerblue', linewidth=0.5, label='$φ_4$')
    ax4.legend(loc="upper right")
    ax5.plot(tk, S[:, 5], color='dodgerblue', linewidth=0.5, label='$φ_5$')
    ax5.legend(loc="upper right")
    fig.text(0.5, 0.005, '$t_k$', ha='center')
    fig.text(0.005, 0.5, '$φ_l(x_k)$', va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig('part_1_eigenfunctions.png')
    plt.show()

def plot_eigenfunctions(S):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3, figsize=(12, 12))
    # Skipping φ0 since it is constant
    ax1.scatter(S[:, 1], S[:, 2], color='dodgerblue', s=2)
    ax1.set_ylabel('$φ_2$')
    ax2.scatter(S[:, 1], S[:, 3], color='dodgerblue', s=2)
    ax2.set_ylabel('$φ_3$')
    ax3.scatter(S[:, 1], S[:, 4], color='dodgerblue', s=2)
    ax3.set_ylabel('$φ_4$')
    ax4.scatter(S[:, 1], S[:, 5], color='dodgerblue', s=2)
    ax4.set_ylabel('$φ_5$')
    ax5.scatter(S[:, 1], S[:, 6], color='dodgerblue', s=2)
    ax5.set_ylabel('$φ_6$')
    ax6.scatter(S[:, 1], S[:, 7], color='dodgerblue', s=2)
    ax6.set_ylabel('$φ_7$')
    ax7.scatter(S[:, 1], S[:, 8], color='dodgerblue', s=2)
    ax7.set_ylabel('$φ_8$')
    ax8.scatter(S[:, 1], S[:, 9], color='dodgerblue', s=2)
    ax8.set_ylabel('$φ_9$')
    ax9.scatter(S[:, 1], S[:, 10], color='dodgerblue', s=2)
    ax9.set_ylabel('$φ_{10}$')
    ax10.scatter(S[:, 1], S[:, 11], color='dodgerblue', s=2)
    ax10.set_ylabel('$φ_{11}$')
    ax11.scatter(S[:, 1], S[:, 12], color='dodgerblue', s=2)
    ax11.set_ylabel('$φ_{12}$')
    ax12.scatter(S[:, 1], S[:, 13], color='dodgerblue', s=2)
    ax12.set_ylabel('$φ_{13}$')
    ax13.scatter(S[:, 1], S[:, 14], color='dodgerblue', s=2)
    ax13.set_ylabel('$φ_{14}$')
    ax14.scatter(S[:, 1], S[:, 15], color='dodgerblue', s=2)
    ax14.set_ylabel('$φ_{15}$')
    ax15.scatter(S[:, 1], S[:, 16], color='dodgerblue', s=2)
    ax15.set_ylabel('$φ_{16}$')

    fig.text(0.5, 0.005, '$φ_1$', ha='center')
    plt.tight_layout()
    # plt.savefig('part_1_eigenfunctions.png')
    plt.show()

    '''
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(12, 12))
'''
def main():
    # Part 1
    N = 1000
    L = 5
    X, tk = generate_points(N)
    tk = np.array(tk)
    #S = diffusion_map_algorithm(X, L)
    #plot_5_eigenfunctions(tk, S)

    # Part 2
    N_1 = 1000
    L_1 = 16
    X_1, t = make_swiss_roll(N_1, noise=0.0, random_state=None)
    S_1 = diffusion_map_algorithm(X_1,L_1)
    print(S_1.shape)
    plot_eigenfunctions(S_1)




if __name__ == '__main__':
    main()
