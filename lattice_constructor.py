# -*- coding: utf-8 -*-
"""
Date: 2025.01.04
Author: Yihan Geng
Group Members: Yangdi Yue, Hongxi Song, Zhining Zhang, Yihan Geng

Description: 
    Python implementation of Algorithm 1 from the paper "Optimization and Identification of Lattice Quantizers".
    This code is developed as part of a group project for the course "Machine Learning".

Paper Reference:
    Erik Agrell, Daniel Pook-Kolb, & Bruce Allen. "Optimization and Identification of Lattice Quantizers." (2024).
    URL: https://arxiv.org/abs/2401.01799
    
Optional Parameters:
    - dim (int, default: 2): The dimension of lattice to be constructed.
    - mode (str, default: 'fast'): Pre-defined parameter combinations ('fast', 'medium', 'slow', 'diy').
    - mu (float, default: 0.005): Initial step size.
    - nu (float, default: 200): Ratio between initial and final step size.
    - step (int, default: 1000000): Number of iteration steps.
    - Tr (int, default: 100): Reduction interval.

Usage:
    Run the script using: `python lattice_constructor.py`
    You can customise parameters using: 
        `python lattice_constructor.py --dim DIM --mode --MODE`
        or `python lattice_constructor.py --dim DIM --mode diy --mu MU --nu NU --step STEP --tr TR`
    The generator matrices over time will be stored in `record_{DIM}_{MODE}.npy`
"""

import numpy as np
import argparse
from tqdm import tqdm


def gram_schmidt(basis):
    """
    对 basis 进行 Gram-Schmidt 正交化
    
    返回正交化后的 B_star 与 Gram-Schmidt 系数 mu, 正交基平方范数 norm
    """
    n, dim = basis.shape
    B_star = np.zeros_like(basis, dtype=float)
    mu = np.zeros((n, n), dtype=float)
    norm = np.zeros(n, dtype=float)

    for i in range(n):
        B_star[i] = basis[i].copy()
        for j in range(i):
            mu[i, j] = np.dot(basis[i], B_star[j]) / norm[j]
            B_star[i] -= mu[i, j] * B_star[j]
        norm[i] = np.dot(B_star[i], B_star[i])
    return B_star, mu, norm


def Red(basis, delta=0.75):
    """
    对 basis 使用 Lenstra–Lenstra–Lovasz algorithm
    delta 为算法中的参数，一般设为 0.75

    返回 reduced basis
    """
    basis = basis.copy().astype(float)
    n, _ = basis.shape
    _, mu, norm = gram_schmidt(basis)

    k = 1
    while k < n:
        # Size reduction
        for j in range(k - 1, -1, -1):
            q = np.round(mu[k, j])
            if q != 0:
                basis[k] -= q * basis[j]
                # Update mu after size reduction
                for l in range(j + 1):
                    mu[k, l] -= q * mu[j, l]

        # Recompute Gram-Schmidt after size reduction
        _, mu, norm = gram_schmidt(basis)

        # Check Lovász condition
        lhs = norm[k]
        rhs = (delta - mu[k, k - 1]**2) * norm[k - 1]
        if lhs >= rhs:
            k += 1
        else:
            # Swap vectors k and k-1
            basis[[k, k - 1]] = basis[[k - 1, k]]
            # Recompute Gram-Schmidt after swap
            _, mu, norm = gram_schmidt(basis)
            k = max(k - 1, 1)

    reduced_basis = basis
    return reduced_basis


def Orth(B):
    '''
    使用 Cholesky 分解将 B 转换为对角线为正的下三角矩阵
    '''
    return np.linalg.cholesky(B @ B.T)


def normalize(B):
    '''
    将 B 关于 V=sqrt(det(BB^T)) 进行归一化
    '''
    n = B.shape[0]
    V = np.prod(np.diag(B))
    return (V**(-1 / n)) * B


def sgn(x):
    '''
    CLP 中使用的辅助函数
    '''
    return -1 if x <= 0 else 1


def CLP(n, G, r):
    '''
    n: dimension
    G: 下三角 lattice generator
    r: 目标点
    找到距离 r 最近的 lattice 点，并返回对应的整系数
    
    参考: Faster recursions in sphere decoding, Algorithm 5
    '''
    # 初始化
    C = np.inf
    i = n
    d = [n - 1 for _ in range(n)]
    lamb = [0 for _ in range(n + 1)]
    F = np.zeros((n, n), dtype=np.float64)
    F[n - 1] = r
    u = np.zeros(n)
    u0 = None
    p = np.zeros(n, dtype=np.float64)
    delta = np.zeros(n, dtype=np.float64)

    # LOOP
    while True:
        while True:
            if i != 0:
                i -= 1
                for j in range(d[i], i, -1):
                    F[j - 1, i] = F[j, i] - u[j] * G[j, i]
                p[i] = F[i, i] / G[i, i]
                u[i] = round(p[i])
                y = (p[i] - u[i]) * G[i, i]
                delta[i] = sgn(y)
                lamb[i] = lamb[i + 1] + y * y
            else:
                u0 = u.copy()
                C = lamb[0]
            if lamb[i] >= C:
                break
        m = i
        while True:
            if i == n - 1:
                return u0
            else:
                i += 1
                u[i] += delta[i]
                delta[i] = -delta[i] - sgn(delta[i])
                y = (p[i] - u[i]) * G[i, i]
                lamb[i] = lamb[i + 1] + y * y
            if lamb[i] < C:
                break

        for j in range(m, i):
            d[j] = i
        for j in range(m - 1, -1, -1):
            if d[j] < i:
                d[j] = i
            else:
                break


def NSM(B, t, rng):
    '''
    使用 Monte-Carlo 计算 lattice generator 对应的 Normalized Second Moment (NSM)
    '''
    n = B.shape[0]
    B = B.copy()
    V = np.prod(np.diag(B))
    norms = np.zeros(t)
    for i in range(t):  # 进行 t 次 sample
        z = rng.random((1, n))
        y = z - CLP(n, B, z @ B)
        e = y @ B
        e = e.reshape(n)
        norms[i] = np.dot(e, e)

    nsm = norms.mean() / (n * (V**(2 / n)))
    return nsm


def train(n, T, Tr, mu, nu, rng):
    '''
    n: dimension
    T: 训练轮数
    Tr: Reduction & Normalize 的时间间隔
    mu: 初始学习率
    nu: 初始与最终学习率的比值
    rng: 随机数生成器
    进行训练，返回生成的 generator matrix
    
    参考: Optimization and Identification of Lattice Quantizers, Algorithm 1
    '''
    # 初始化
    B = np.random.randn(n, n)  # GRAN, 据说 numpy 使用 Box-Muller 实现
    B = Red(B)  # RED, 使用 Lenstra–Lenstra–Lovasz algorithm
    B = Orth(B)  # ORTH, 使用 Cholesky 分解将 B 转换为对角线为正的下三角矩阵
    B = normalize(B)  # 关于 V 归一化
    recorder = []
    # 开始训练
    for t in tqdm(range(T)):
        if (t + 1) % 10000 == 0:
            recorder.append(B)
        z = rng.random((1, n))
        y = z - CLP(n, B, z @ B)
        e = y @ B
        y = y.reshape(n)
        e = e.reshape(n)
        norm_e = np.dot(e, e)
        for i in range(n):
            B[i, :i] -= mu * y[i] * e[:i]
            B[i, i] -= mu * (y[i] * e[i] - norm_e / (n * B[i, i]))
        if t % Tr == Tr - 1:
            B = normalize(Orth(Red(B)))
            if t % (100 * Tr) == 100 * Tr - 1:
                print(f'step: {t + 1}/{T}, NSM: {NSM(B, 1000, rng)}')  # 阶段性输出 NSM，使用较少 sample 以加快速度
        mu *= nu

    recorder.append(B)
    recorder = np.array(recorder)
    print(recorder.shape[0])
    return B, recorder


def main():
    parser = argparse.ArgumentParser(description="Lattice Constructor")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--mode", type=str, default='fast')
    parser.add_argument("--mu", type=float, default=0.005)
    parser.add_argument("--nu", type=float, default=200)
    parser.add_argument("--step", type=int, default=1000000)
    parser.add_argument("--tr", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 设置参数
    np.random.seed(args.seed)
    n = args.dim
    if args.mode == 'fast':
        T = 1000000
        Tr = 100
        mu = 0.005
        nu = 200**(-1 / (T - 1))
    elif args.mode == 'medium':
        T = 10000000
        Tr = 100
        mu = 0.001
        nu = 500**(-1 / (T - 1))
    elif args.mode == 'slow':
        T = 100000000
        Tr = 100
        mu = 0.0005
        nu = 1000**(-1 / (T - 1))
    else:
        T = args.step
        Tr = args.tr
        mu = args.mu
        nu = args.nu**(-1 / (T - 1))
    rng = np.random.default_rng(seed=args.seed)  # URAN, 使用 permuted congruential generator

    # recorder_sequence = [0.01, 0.03, 0.1]
    B, recorder = train(n, T, Tr, mu, nu, rng)

    print(f"Constructed generator matrix for dimension {n}:\n{B}")
    print(f"Normalized Second Moment (NSM): {NSM(B, 1000000, rng):.6f}")  # 计算最终的 NSM

    np.save(f'record_{args.dim}_{args.mode}.npy', recorder)


if __name__ == "__main__":
    main()