"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 18:51:52
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : TOPSIS.py
# @Software: PyCharm
-------------------------------------
"""

import pandas as pd
import numpy as np


def entropy_weight(features):
    """

    Entropy method

    Args:
        features: Features

    Returns:
        weight: weight coefficient

    """
    features = np.array(features)
    proportion = features / features.sum(axis=0)  # normalized
    entropy = np.nansum(-proportion * np.log(proportion) / np.log(len(features)),
                        axis=0)  # calculate entropy
    weight = (1 - entropy) / (1 - entropy).sum()

    return weight  # calculation weight coefficient


def topsis(data, weight=None):
    """

    TOPSIS algorithm

    Args:
        data: Features
        weight:

    Returns:
        Result:
        Z:
        weight:

    """
    # normalized
    data = data / np.sqrt((data ** 2).sum())
    # best and worst solution
    Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])

    weight = entropy_weight(data) if weight is None else np.array(weight)  # distance
    Result = data.copy()
    Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
    Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))

    # composite score index
    Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
    Result['排序'] = Result.rank(ascending=False)['综合得分指数']

    return Result, Z, weight
