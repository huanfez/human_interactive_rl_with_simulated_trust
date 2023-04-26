#! usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, invgamma, norm
from numpy.linalg import inv
from numpy.core._multiarray_umath import ndarray
from math import log, exp


def logprob(y, x, A):
    sum = 0
    for j in range(0,1):
        sum = sum + exp(A[j] * x)

    if y == 1:
        return -log(1 + sum)
    else:
        return A[y] * x - log(1 + sum)


def logXposterior(Y, Xm, X, Z, Beta, sigma_square, A):
    length = len(Y)
    logsum = 0
    for k in range(0, length):
        logsum = logsum + logprob(Y[k], Xm[k], A) + multivariate_normal.logpdf(X[k], np.matmul(Beta.T,Z[k]), sigma_square)

    return logsum


pl1 = plt.subplot(251)
pl2 = plt.subplot(252)
pl3 = plt.subplot(253)
pl4 = plt.subplot(254)
pl5 = plt.subplot(255)
pl6 = plt.subplot(256)
pl7 = plt.subplot(257)
pl8 = plt.subplot(258)
pl9 = plt.subplot(259)
pl10 = plt.subplot(2,5,10)


## source data
def ratePr(level, xvalue, Avector):
    expsum = 1
    for avalue in Avector:
        expsum = expsum + exp(xvalue*avalue)

    if level == 1:
        return 1 / expsum
    else:
        return exp(xvalue * Avector[level]) / expsum

# Z1_K = np.array([[0.5,0.5], [0.6,0.4], [0.8,0.7], [0.75,0.9], [0.6,0.9], [0.4,0.5], [0.9,0.4], [0.7,0.6]])
# Z2_K = np.array([[0.5,0.48], [0.5,0.5], [0.66,0.75], [0.7,0.8], [0.68,0.95], [0.5,0.54], [0.88,0.3], [0.8,0.5]])
# Z3_K = np.array([[0.52,0.48], [0.7,0.6], [0.75,0.7], [0.72,0.88], [0.7,0.86], [0.6,0.42], [0.72,0.5], [0.78,0.6]])
K = 10

Z1_K = np.random.rand(K, 3)
Z1_K[:, 2] = 1.0

Z2_K = Z1_K + np.random.rand(K, 3) * 0.1
Z2_K[:, 2] = 1.0

Z3_K = Z1_K - np.random.rand(K, 3) * 0.1
Z3_K[:, 2] = 1.0

betas = np.array([0.3, 0.7, 0.5])
xi = 0.1
x1_true = np.matmul(Z1_K,betas.T) + np.random.rand(K) * xi
x2_true = np.matmul(Z2_K,betas.T) + np.random.rand(K) * xi
x3_true = np.matmul(Z3_K,betas.T) + np.random.rand(K) * xi

## organized (additional) data
Z_3K = np.concatenate((Z1_K, Z2_K, Z3_K))
# insert X (as a column) into Z_3K
Ztilde_3K = Z_3K
x3K_s = np.concatenate((x1_true, x2_true, x3_true))

""" Hyperparameter prior """
# Beta: mutivariate normal
Mu0 = np.array([1.1,1.1, 1.1])  # prior
Cov0 = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])  # prior
Beta_s = (np.random.multivariate_normal(Mu0.flatten().tolist(), Cov0.tolist(), size=None)).T  # initial sampling

# residue: inverse gamma
nu0 = 3.0  # prior
sigma0 = 3.0  # prior
eta_s = np.random.gamma(nu0, sigma0, size=None) # initial sampling
sigma_square_s = 1.0 / eta_s

""" MCMC: Gibbs sampling """
for itr in range (0, 3000):
    # Beta: mutivariate normal
    Ztilde_3K_tran = Ztilde_3K.T
    # x3K = np.matmul(Ztilde_3K_tran, Beta_s)
    Cov3K = inv(inv(Cov0) + np.matmul(Ztilde_3K_tran, Ztilde_3K) * eta_s)  # posterior
    Mu3K_ = np.matmul(inv(Cov0), Mu0) + np.matmul(Ztilde_3K_tran, x3K_s) * eta_s
    Mu3K = np.matmul(Cov3K, Mu3K_)  # posterior
    # Beta_posterior = multivariate_normal(Mu3K.flatten().tolist(), Cov3K.tolist())
    Beta_s = (np.random.multivariate_normal(Mu3K.flatten().tolist(), Cov3K.tolist(), size=None)).T  # Gibbs sampling

    # residue: inverse gamma
    error3K_s = x3K_s - (np.matmul(Ztilde_3K, Beta_s))  # vector operation
    sigma3K = (nu0 * sigma0 * sigma0 + np.matmul(error3K_s.T, error3K_s)) / 2.0  # posterior
    nu3K = (nu0 + 3.0 * K) / 2.0  # posterior
    eta_s = np.random.gamma(nu3K, sigma3K, size=None)
    sigma_square_s = 1.0 / eta_s

    pl1.plot(Beta_s[0], Beta_s[1],'.')
    pl2.plot(itr, Beta_s[0], '.')
    pl3.plot(itr, Beta_s[1], '.')
    pl4.plot(itr, Beta_s[2], '.')
    pl5.plot(itr, 1.0/eta_s, '.')

    # pl9.plot(itr, Alpha_s[1], '.')
    # pl10.plot(itr, Alpha_s[2], '.')
    # weight.contourf(w1, w2, posterior.pdf(pos))
    # plt.pause(0.001)
    plt.draw()

pl2.plot(range(0,3000), np.ones(3000)* betas[0], '-')
pl3.plot(range(0,3000), np.ones(3000)* betas[1], '-')
pl4.plot(range(0,3000), np.ones(3000)* betas[2], '-')

plt.show()
