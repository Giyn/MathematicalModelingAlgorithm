"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 18:51:52
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : FCEM.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np


def min_max_operator(A, R):
    """

    combining matrices with the most value operator

    Args:
        A: judging factor weight vector
        R: fuzzy relation matrix

    Returns:
        weight: weight coefficient

    """
    B = np.zeros((1, R.shape[1]))

    for column in range(0, R.shape[1]):
        list_ = []
        for row in range(0, R.shape[0]):
            list_.append(min(A[row], R[row, column]))
        B[0, column] = max(list_)

    return B


def mul_max_operator(A, R):
    """

    synthesize matrices with multiplicative maximum operator

    Args:
        A: judging factor weight vector
        R: fuzzy relation matrix

    Returns:
        weight: weight coefficient

    """
    B = np.zeros((1, R.shape[1]))

    for column in range(0, R.shape[1]):
        list_ = []
        for row in range(0, R.shape[0]):
            list_.append(A[row] * R[row, column])
        B[0, column] = max(list_)

    return B


def mymin(list_):
    for index in range(1, len(list_)):
        if index == 1:
            temp = min(1, list_[0] + list[1])
        else:
            temp = min(1, temp + list_[index])

    return temp


def min_mymin_operator(A, R):
    """

    use the minimum minimum operator to synthesize matrix

    Args:
        A: judging factor weight vector
        R: fuzzy relation matrix

    Returns:
        weight: weight coefficient

    """
    B = np.zeros((1, R.shape[1]))

    for column in range(0, R.shape[1]):
        list_ = []
        for row in range(0, R.shape[0]):
            list_.append(min(A[row], R[row, column]))
        B[0, column] = mymin(list_)

    return B


def mul_mymin_operator(A, R):
    """

    synthesize matrices with multiplication minimum operator

    Args:
        A: judging factor weight vector
        R: fuzzy relation matrix

    Returns:
        weight: weight coefficient

    """
    B = np.zeros((1, R.shape[1]))

    for column in range(0, R.shape[1]):
        list_ = []
        for row in range(0, R.shape[0]):
            list_.append(A[row] * R[row, column])
        B[0, column] = mymin(list_)

    return B
