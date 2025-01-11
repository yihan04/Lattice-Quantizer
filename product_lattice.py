# -*- coding: utf-8 -*-
"""
Date: 2025.01.10
Author: Yihan Geng, Zhining Zhang
Group Members: Yangdi Yue, Hongxi Song, Zhining Zhang, Yihan Geng

Description: 
    Python implementation of NSM calculation of product lattice in the paper "On the Best Lattice Quantizers".
    This code is developed as part of a group project for the course "Machine Learning".
    
Paper Reference:
    Erik Agrell, & Bruce Allen. "On the Best Lattice Quantizers." (2023).
    URL: http://dx.doi.org/10.1109/TIT.2023.3291313

Usage:
    Run the script using: `python product_lattice.py`
"""


def product_nsm(nsm_1, dim_1, nsm_2, dim_2):
    '''
    Calculate the NSM of the optimal product lattice constructed from two lattices.
    '''
    return ((nsm_1**dim_1) * (nsm_2**dim_2))**(1 / (dim_1 + dim_2))


def get_sota():
    '''
    汇集目前所有 SOTA 值
    '''
    sota = [
        0.083333333, 0.080187537, 0.078543281, 0.076603235, 0.075625443, 0.074243697, 0.073116493, 0.071682099, 0.071622594,
        0.070813818, 0.070426259, 0.070031226, 0.071034583, 0.06952, 0.068871726, 0.068297622, 0.06910, 0.06866, 0.06936,
        0.06988, 0.06998, 0.06853, 0.06912, 0.06941, 0.06640, 0.06678, 0.06708, 0.06722, 0.06737, 0.06738, 0.06736, 0.06720,
        0.06732, 0.06722, 0.06720, 0.06718, 0.06757, 0.06781, 0.06799, 0.06677, 0.06713, 0.06736, 0.06753, 0.06761, 0.06770,
        0.06770, 0.06768, 0.06577
    ]

    return sota


def main():
    best_nsm = get_sota()  # 初始值
    reference = [[i + 1] for i in range(len(best_nsm))]  # 每个 lattice 的构成
    prod_reference = [[i + 1] for i in range(len(best_nsm))]
    prod_nsm = [1 for _ in range(len(best_nsm))]  # product lattice 的值
    new = []  # 超过 sota 的维度

    for i in range(len(best_nsm)):
        if i == 0:
            continue
        for j in range((i + 1) // 2):
            prod = product_nsm(nsm_1=best_nsm[j], dim_1=j + 1, nsm_2=best_nsm[i - j - 1], dim_2=i - j)
            prod = round(prod, 9)
            if prod < prod_nsm[i]:
                prod_nsm[i] = prod
                prod_reference[i] = reference[j] + reference[i - j - 1]
        if prod_nsm[i] < best_nsm[i]:
            best_nsm[i] = prod_nsm[i]
            reference[i] = prod_reference[i] + []
            new.append(i + 1)

    print(f'Better lattices are constructed in dimensions {new}')
    print(f'NSM of product lattices:\n{prod_nsm}')
    print(f'Construction of product lattices:\n{prod_reference}')


if __name__ == "__main__":
    main()