#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of diagnostic tools to assess convergence.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

import matplotlib.pyplot as plt
import numpy as np


def gelman_rubin(state_dict):
    raise NotImplementedError


def plot_gelman_rubin_brooks(state_dict):
    raise NotImplementedError


def plot_likelihood(state_dict, figsize=(12, 5), n_chains=240):
    """Plot the distribution and evolution of logP values.

    Parameters
    ----------
    state_dict : gaps state blob
        The state of the sampler as returned by `gaps.run_sampler`.
        Currently only ``logP` and `flatlogP` are used by this function.
    figsize : tuple of ints
        Size of the figure. This figure has 2 columns and 1 row.
        Default: (12, 5).
    n_chains : int
        Number of chains to be used to plot the evolution of logP
        values. Default: 240.

    Returns
    -------
    None
    """
    logP = state_dict['logP']
    flatlogP = state_dict['flatlogP']
    n_walkers = logP.shape[0]
    selection = np.random.permutation(n_walkers)[:n_chains]
    logPmin, logPmax = np.percentile(flatlogP, [0.1, 100])
    logPmax += (logPmax - logPmin) / 24

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    for walker in logP[selection]:
        plt.plot(walker, 'k', alpha=0.01)
    plt.xlabel('Step Index')
    plt.ylabel('logP')
    plt.ylim(logPmin, logPmax)

    plt.subplot(1, 2, 2)
    plt.hist(flatlogP[flatlogP > logPmin], bins=120)
    plt.xlabel('logP')
    plt.ylabel('Count')
    plt.xlim(logPmin, logPmax)

    plt.tight_layout()
    return


if __name__ == '__main__':
    pass
