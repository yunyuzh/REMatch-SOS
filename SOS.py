import os
import numpy as np
import pandas as pd

def load_distance_matrix(file_path):
    return np.load(file_path)

def d2a(D, perplexity=30, eps=1e-5):
    (n, _) = D.shape
    A = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax =  np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisA) = get_perplexity(Di, beta[i])

        Hdiff = H - logU
        tries = 0
        while (np.isnan(Hdiff) or np.abs(Hdiff) > eps) and tries < 5000:
            if np.isnan(Hdiff):
                beta[i] = beta[i] / 10.0
            elif Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0
            (H, thisA) = get_perplexity(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        A[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisA

    return A

def a2b(A):
    B = A / A.sum(axis=1)[:,np.newaxis]
    return B

def b2o(B):
    O = np.prod(1-B, 0)
    return O

def get_perplexity(D, beta):
    A = np.exp(-D * beta)
    sumA = np.sum(A)
    H = np.log(sumA) + beta * np.sum(D * A) / sumA
    return H, A