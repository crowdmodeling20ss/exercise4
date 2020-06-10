import math

import numpy as np
import scipy
import scipy.spatial
from scipy.linalg import eigh


def generate_points(N):
    """
    Generate periodic data set

    :param N:
    :return: [N, 2] numpy array
    """
    # X = {xk ∈ R2}Nk=1, xk = (cos(tk), sin(tk)), tk = (2πk)/(N + 1).
    data = []
    for k in range(1, N + 1):
        t_k = (2 * math.pi * k) / (N + 1)  # tk = (2πk)/(N + 1).
        x_k = (math.cos(t_k), math.sin(t_k))
        data.append(x_k)
    X = np.array(data)

    return X


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


def main():
    N = 1000
    L = 5
    X = generate_points(N)
    # X = np.array([[1, 2], [3, 4], [5, 6]])

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
    K = P_inverse.dot(P_inverse).dot(W).dot(P_inverse)
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
    # 9. Compute the eigenvalues of Tˆ1/ε by λ2 = a1/ε.
    lambda_l = a_l ** (1 / epsilon)  # TODO: Do we need to take square root of a_l?
    # 10. Compute the eigenvectors φl of the matrix T = Q−1K by φl = Q−1/2vl.
    S = np.matmul(Q_inverse, v_l.T)


if __name__ == '__main__':
    main()
