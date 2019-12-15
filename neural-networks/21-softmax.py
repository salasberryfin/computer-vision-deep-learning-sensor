import numpy as np

def softmax(L):
    exps = np.exp(L)
    values = []
    for l in exps:
        values.append(l / np.sum(exps))

    return values

