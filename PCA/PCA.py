"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 23:04:45
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : PCA.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np
import pandas as pd


def pca(X, k):
    """

    principal components analysis

    Args:
        X:
        k: the components you want

    Returns:
        new data

    """
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))

    return data


if __name__ == '__main__':
    X_column = range(1, 14)
    df = pd.read_csv('wine.csv').iloc[:, X_column].values
    print(df)
    new_df = pca(df, 6)
    print(new_df)
