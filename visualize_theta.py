# -*- coding: utf-8 -*-
"""
Date: 2025.01.10
Author: Zhining Zhang, Yihan Geng
Group Members: Yangdi Yue, Hongxi Song, Zhining Zhang, Yihan Geng

Description: 
    Python implementation of theta image from the paper "Optimization and Identification of Lattice Quantizers".
    Implemetned through trivial modifications to the kissing number algorithm in the paper "Closest Point Search in Lattices".
    This code is developed as part of a group project for the course "Machine Learning".
    
Paper Reference:
    Erik Agrell, Daniel Pook-Kolb, & Bruce Allen. "Optimization and Identification of Lattice Quantizers." (2024).
    URL: https://arxiv.org/abs/2401.01799
    Erik Agrell, Thomas Eriksson, Alexander Vardy, & Kenneth Zeger. "Closest Point Search in Lattices." (2002).
    URL: https://ieeexplore.ieee.org/document/1019833
    
Optional Parameters:
    - dim (int, default: 2): The dimension of the lattice constructed.
    - mode (str, default: 'fast'): Pre-defined parameter combinations ('fast', 'medium', 'slow', 'diy').

Usage:
    Run `python lattice_constructor.py --dim DIM --mode MODE` to get generator matrices in `record_{DIM}_{MODE}.npy`
    Then you can run `visualize_theta.py --dim DIM --mode MODE` to get the theta image.
"""

import argparse
import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt
from lattice_constructor import normalize, Orth, Red, sgn


def ThetaNumber(G, r):
    """
    Compute the theta number with regard to r.
    Modified from the kissing number algorithm of "Closest Point Search in Lattices."

    Parameters:
    G (ndarray): An n x m generator matrix.
    """
    # G_2 := WG, where W is an n*n unimodular matrix (identity here)
    n = G.shape[0]
    W = np.eye(n, dtype=int)
    G_2 = np.matmul(W, G)

    # G_2 = G_3Q, Q is an orthonormal matrix,
    # and G_3 is an n*n lower-triangular matrix with positive diagonal elements.
    QT, G_3T = qr(G_2.T)  # QR decomposition
    Q = QT.T
    G_3 = G_3T.T
    for i in range(min(G_3.shape)):
        if G_3[i, i] < 0:
            G_3[:, i] *= -1
            Q[i, :] *= -1

    H_3 = np.linalg.inv(G_3)  # H_3 := G_3^(-1).
    u_3 = decode(H_3, r)

    # 统计在 r 范围内 lattice 点的个数
    val = [np.linalg.norm(np.matmul(u_3[:, k], G_2).reshape(n), ord=2) for k in range(u_3.shape[1])]

    return len([x for x in val if x <= r])


def decode(H, r):
    """
    Decodes the vector 0 in the lattice A(H^(-1)) using the DECODE algorithm.
    Returns the coefficients of all the lattice points within the distance r.

    Parameters:
    H (numpy.ndarray): An n x n lower-triangular matrix with positive diagonal elements.
    """
    n = H.shape[0]
    bound = r**2  # distance bound
    k = n  # Dimension of examined layer
    dist = np.zeros(n + 1)  # Distance to examined layer
    e = np.zeros((n + 1, n))  # Used to compute u_k
    u = np.zeros(n + 1, dtype=int)  # Examined lattice point
    step = np.zeros(n + 1)  # Offset to next layer
    best_u = np.zeros((n, 0))

    # Initializing e[k] as 0
    e[k, :] = np.zeros(n)

    # Initial calculations for the top layer
    u[k] = np.round(e[k, k - 1])
    y = (e[k, k - 1] - u[k]) / H[k - 1, k - 1]  # Residual
    step[k] = sgn(y)  # Step direction

    while True:
        newdist = dist[k] + y**2

        if newdist < bound * (1 + 1e-12):
            if k != 1:
                # Move down a layer
                for i in range(k - 1):
                    e[k - 1, i] = e[k, i] - y * H[k - 1, i]
                k -= 1
                dist[k] = newdist
                u[k] = np.round(e[k, k - 1])
                y = (e[k, k - 1] - u[k]) / H[k - 1, k - 1]
                step[k] = sgn(y)
            else:
                # Update the best lattice point
                best_u = np.concatenate((best_u, u[1:n + 1].copy().reshape(n, 1)), axis=1)
                # Move up a layer
                u[k] += step[k]
                y = (e[k, k - 1] - u[k]) / H[k - 1, k - 1]
                step[k] = -step[k] - sgn(step[k])
        else:
            if k == n:
                return best_u
            else:
                # Move up a layer
                k += 1
                u[k] += step[k]
                y = (e[k, k - 1] - u[k]) / H[k - 1, k - 1]
                step[k] = -step[k] - sgn(step[k])


def visualize(data, times, dim, mode):
    """
    进行可视化，生成 theta image
    """

    plt.rcParams['font.family'] = 'serif'

    plt.figure(figsize=(8, 6))
    alphas = [0.5, 0.5, 0.7, 0.9, 1.0]
    for i, now_d in enumerate(data):
        x = [point[0] for point in now_d]
        y = [point[1] for point in now_d]
        c = 10000
        if mode == 'slow':
            c *= 10
        plt.plot(x, y, label=f"$t={times[i] * c}$", alpha=alphas[i], linestyle='--' if i != len(data) - 1 else '-')

    # plt.plot(x, y, color='blue', label=r'Data Line')

    plt.title(r'Dimension ' + f'{dim}', fontsize=16)
    plt.xlabel(r'$r^2$', fontsize=14)
    plt.ylabel(r'$N(B, r)$', fontsize=14)
    plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'dim{dim}_mode{mode}.jpg', dpi=1000)
    # plt.show()


def get_array():
    '''
    十五维已知最好 lattice 的 generator matrix
    '''
    B = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [
                      1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2,
                      1 / 2, 1 / 2
                  ]])

    return normalize(Orth(Red(B)))


def main():
    parser = argparse.ArgumentParser(description="Theta Image")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--mode", type=str, default='fast')
    args = parser.parse_args()
    dim = args.dim
    mode = args.mode
    Bs = np.load(f'record_{dim}_{mode}.npy')

    all_data = []
    times = [0, 1, 3, 10, 100]  # 选取不同时间点的 generator matrix
    for ind in times:
        print(ind)
        B = Bs[ind]
        data = []
        # B = get_array()
        for r in np.arange(0.0, 5.5, 0.01):
            print(r)
            data.append((r, ThetaNumber(B, r**0.5)))
        # if ind == 100:
        #     B = get_array()
        #     for r in np.arange(0.0, 5.5, 0.01):
        #         print(r)
        #         data.append((r, ThetaNumber(B, r**0.5)))
        # else:
        #     for r in np.arange(0.0, 5.5, 0.01):
        #         data.append((r, 1))
        all_data.append(data)
    visualize(all_data, times, dim, mode)


if __name__ == "__main__":
    main()