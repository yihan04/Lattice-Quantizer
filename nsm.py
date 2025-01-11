# -*- coding: utf-8 -*-
"""
Date: 2025.01.10
Author: Yihan Geng
Group Members: Yangdi Yue, Hongxi Song, Zhining Zhang, Yihan Geng

Description: 
    Python implementation of NSM calculation using Monte-Carlo.
    This code is developed as part of a group project for the course "Machine Learning".

Usage:
    Run the script using: `python nsm.py`
    You can customise parameters using: 
        `python nsm.py --dim DIM --mode MODE --step STEP`
"""

from lattice_constructor import NSM
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="NSM Calculation")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--mode", type=str, default='fast')
    parser.add_argument("--step", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 设置参数
    np.random.seed(args.seed)
    dim = args.dim
    t = args.step
    mode = args.mode
    rng = np.random.default_rng(seed=args.seed)  # URAN, 使用 permuted congruential generator
    Bs = np.load(f'record_{dim}_{mode}.npy')
    B = Bs[100]
    print(f"Normalized Second Moment (NSM) of dim {dim}: {NSM(B, t, rng):.6f}")  # 计算最终的 NSM


if __name__ == "__main__":
    main()