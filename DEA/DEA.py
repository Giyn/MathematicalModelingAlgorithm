"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2021/1/21 22:06:09
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : DEA.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np
from scipy.optimize import fmin_slsqp


class DEA(object):
    def __init__(self, inputs, outputs):
        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]

        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas

    def __efficiency(self, unit):
        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)

        return (numerator / denominator)[unit]

    def __target(self, x, unit):
        # unroll the weights
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]

        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)

        return numerator / denominator

    def __constraints(self, x, unit):
        # unroll the weights
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m + self.r)], x[(self.m + self.r):]
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t * self.inputs[unit, input] - lhs
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])

        return np.array(constr)

    def __optimize(self):
        d0 = self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,))
            # unroll weights
            self.input_w = x0[:self.m]
            self.output_w = x0[self.m:(self.m + self.r)]
            self.lambdas = x0[(self.m + self.r):]

            self.efficiency[unit] = self.__efficiency(unit)

    def fit(self):
        self.__optimize()  # optimize
        return self.efficiency


if __name__ == "__main__":
    X = np.array([
        [20., 300.],
        [30., 200.],
        [40., 100.],
        [20., 200.],
        [10., 400.]])
    y = np.array([
        [1000.],
        [1000.],
        [1000.],
        [1000.],
        [1000.]])
    dea = DEA(X, y)
    rs = dea.fit()
    print(rs)
