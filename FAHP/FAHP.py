"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/31 11:55:46
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : FAHP.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np


def fahp(matrix: np.array) -> list:
    N = matrix.shape[0]
    A1 = []
    R1 = []
    W = []
    R = np.zeros((N, N), dtype=np.float)
    SUM = 0

    for i in range(N):
        A1.append(0)  # 记录模糊互补矩阵每行和
        R1.append(0)  # 记录一致性矩阵每行积
        W.append(0)

    for i in range(N):
        A1[i] = 0  # 记录每行的和
        for j in range(N):
            A1[i] += matrix[i][j]

    # 转换成模糊一致性矩阵
    for i in range(N):
        for j in range(N):
            R[i][j] = (A1[i] - A1[j]) / (2 * N) + 0.5

    # 幂积法求单层权重
    for i in range(N):
        R1[i] = 1
        for j in range(N):
            R1[i] *= R[i][j]
        W[i] = pow(R1[i], 0.2)
        SUM += W[i]

    for i in range(N):
        W[i] = W[i] / SUM

    return W


if __name__ == '__main__':
    # fuzzy complementary matrix
    M = np.array([
        [0.50, 0.75, 0.80, 0.60, 0.50, 0.55],
        [0.25, 0.50, 0.65, 0.50, 0.30, 0.40],
        [0.20, 0.35, 0.50, 0.40, 0.30, 0.40],
        [0.40, 0.50, 0.60, 0.50, 0.30, 0.40],
        [0.50, 0.70, 0.70, 0.70, 0.50, 0.75],
        [0.45, 0.60, 0.60, 0.60, 0.25, 0.50]
    ])
    print(fahp(M))
