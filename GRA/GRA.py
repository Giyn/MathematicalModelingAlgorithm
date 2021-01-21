"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 16:23:28
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : GRA.py
# @Software: PyCharm
-------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def dimensionless(df):
    """

    dimensionless

    Args:
        df: DataFrame

    Returns:
        new DataFrame

    """
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        MEAN = d.mean()
        newDataFrame[c] = ((d - MEAN) / (MAX - MIN)).tolist()

    return newDataFrame


def gra_one(gray, m=0):
    """

    calculate correlation coefficient of one column

    Args:
        gray:
        m   : reference series

    Returns:
        correlation coefficient of one column

    """
    gray = dimensionless(gray)
    # 参考数列
    reference = gray.iloc[:, m]
    gray.drop(str(m), axis=1, inplace=True)
    # 比较数列
    compare = gray.iloc[:, 0:]
    # 计算行列
    shape_n = compare.shape[0]
    shape_m = compare.shape[1]

    # 与参考数列比较, 相减
    a = np.zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            a[i, j] = abs(compare.iloc[j, i] - reference[j])

    max_ = np.amax(a)
    min_ = np.amin(a)

    # 计算值
    result = np.zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            result[i, j] = (min_ + 0.5 * max_) / (a[i, j] + 0.5 * max_)

    # 求均值, 得到灰色关联值
    result_list = [np.mean(result[i, :]) for i in range(shape_m)]
    result_list.insert(m, 1)

    return pd.DataFrame(result_list)


def gra(dataframe):
    """

    calculate correlation coefficient of a dataframe

    Args:
        dataframe: dataframe

    Returns:
        correlation coefficient of a dataframe

    """
    df = dataframe.copy()
    list_columns = [str(s) for s in range(len(df.columns)) if s not in [None]]
    df_local = pd.DataFrame(columns=list_columns)
    df.columns = list_columns

    for i in range(len(df.columns)):
        df_local.iloc[:, i] = gra_one(df, m=i)[0]

    return df_local


def show_gra_heatmap(df):
    """

    show heatmap of gray relational analysis

    Args:
        df: dataframe

    """
    colormap = plt.cm.RdBu
    y_labels = df.columns.values.tolist()
    f, ax = plt.subplots(figsize=(14, 14))
    ax.set_title('GRA HeatMap')

    with sns.axes_style("white"):
        sns.heatmap(df, cmap="YlGnBu", annot=True)

    plt.savefig('HeatMap.png')
    plt.show()


if __name__ == '__main__':
    wine = pd.read_csv("wine.csv")

    data_wine_gra = gra(wine)
    data_wine_gra.to_csv("GRA.csv")

    data_wine_gra.columns = wine.columns
    data_wine_gra.index = wine.columns

    show_gra_heatmap(data_wine_gra)
