import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    values = []
    for i in range(len(Y)):
        values.append(Y[i]*np.log(P[i]) + (1-Y[i])*np.log(1-P[i]))

    return -np.sum(values)


if __name__ == "__main__":
    Y = [1, 1, 0]
    P = [0.8, 0.7, 0.1]
    cross_entropy(Y, P)
