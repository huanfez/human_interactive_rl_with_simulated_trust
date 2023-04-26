#! usr/bin/env python2

import numpy as np
from numpy.linalg import inv


def iterated_sampling(Z_3_K, x_3_K, Mu0=np.array([1.1, 1.1, 1.1]), Cov0=np.diag([0.5, 0.5, 0.5]), nu0=3.0, sigma0=3.0,
                      iters=5000):
    """
    Parameters:
        Z_3_K: ndarray - input data
        x_3_K: ndarray - output data
        Mu0: ndarray - prior mean of Beta
        Cov0: ndarray - prior variance of Beta
        nu0: scalar - prior hyperparameters of residue
        sigma0: scalar - prior hyperparameters of residue
        iters: iterations
    Returns: posterior hyperparameters of beta and residue
    """
    samples_Beta = []
    samples_sigma_square = []

    eta_s = np.random.gamma(nu0, sigma0, size=None)  # initial sampling of residue
    sigma_square_s = 1.0 / eta_s

    Z_3K = np.concatenate((Z_3_K[:, 0, :], Z_3_K[:, 1, :], Z_3_K[:, 2, :]))
    x3K = x_3_K.flatten(order='F')

    for itr in range(0, iters):
        Cov3K = inv(inv(Cov0) + np.matmul(Z_3K.T, Z_3K) * eta_s)  # posterior
        Mu3K_ = np.matmul(inv(Cov0), Mu0) + np.matmul(Z_3K.T, x3K) * eta_s
        Mu3K = np.matmul(Cov3K, Mu3K_)  # posterior
        Beta_s = (np.random.multivariate_normal(Mu3K.flatten().tolist(), Cov3K.tolist(), size=None)).T  # Gibbs sampling
        samples_Beta.append(Beta_s)

        # residue: inverse gamma
        error3K_s = x3K - (np.matmul(Z_3K, Beta_s))  # vector operation
        sigma3K = (nu0 * sigma0 * sigma0 + np.matmul(error3K_s.T, error3K_s)) / 2.0  # posterior
        nu3K = (nu0 + 3.0 * len(x3K)) / 2.0  # posterior
        eta_s = np.random.gamma(nu3K, sigma3K, size=None)
        sigma_square_s = 1.0 / eta_s
        samples_sigma_square.append(sigma_square_s)

    return samples_Beta, samples_sigma_square


def mean_value_model_parameters(samples_Beta, samples_sigma_square):
    # traces of posterior Beta
    means_Beta = np.mean(samples_Beta[1000:], axis=0)
    variance_Beta = np.cov(np.asarray(samples_Beta[1000:]).T)

    means_sigma_square = np.mean(samples_sigma_square[1000:], axis=0)
    variance_sigma_square = np.var(samples_sigma_square[1000:], axis=0)

    return means_Beta, variance_Beta, means_sigma_square, variance_sigma_square
