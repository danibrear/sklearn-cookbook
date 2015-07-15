import numpy as np

# how MSE works:
# Takes each predictions deviance from the actual value, squares it, then
# averages the squared values. The Gauss-Markov theorem guarantees that the
# solution to linear regression is the best in the sense that the
# coefficients have the smallest expected squared error.
def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)

# How MAD works:
# MAD is similar to MSE in that it averages to get the solution. The only
# differnce is that MAD is the absolute value of the difference between
# the prediction and the actual value.
def MAD(target, predictions):
    absolute_deviation = np.abs(target - predictions)
    return np.mean(absolute_deviation)
    