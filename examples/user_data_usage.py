# -*- coding: utf-8 -*-
"""
Examples to illustrate the basic use of GAPS.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

from emcee import EnsembleSampler
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np

# There is no installation functionality yet. To use the module, we
# add the path manually for now.
import sys
sys.path.append(r'/path/to/GAPS')
sys.path.append(r'C:\Files\Dropbox\Python\GAPS')
import gaps


# For comparison, define some distribution in Python.
def logP(point, data):
    mean, log_sigma = point
    if not ((-5 < mean < 5) and (-1 < log_sigma < 0)):
        return -np.inf
    sigma = np.exp(log_sigma)
    data = np.array(data).flatten()
    return np.sum(norm.logpdf(data, loc=mean, scale=sigma))


data = norm.rvs(loc=1, scale=0.6, size=12)

mean_axis = np.linspace(-5, 5, 240)
log_sigma_axis = np.linspace(-1, 0, 240)
mean, log_sigma = np.meshgrid(mean_axis, log_sigma_axis, indexing='ij')
z = np.empty_like(mean)
for i, j in np.ndindex(240, 240):
    z[i, j] = np.exp(logP((mean[i, j], log_sigma[i, j]), data))
plt.figure()
plt.pcolormesh(mean, log_sigma, z)


# As an additional comparison, use emcee to sample the distribution.
n_walkers, n_steps, n_dim = 120, 600, 2
initial_state = np.random.uniform([-5, -1], [5, 0], size=(n_walkers, n_dim))
sampler = EnsembleSampler(n_walkers, n_dim, logP, args=(data,))
sampler.run_mcmc(initial_state, n_steps)
plt.figure()
plt.hist2d(sampler.flatchain[:, 0], sampler.flatchain[:, 1], bins=64)


# Define the logP function in OpenCL code.
# cfloat is the medium precision float type of GAPS.
# User data is currently always converted to cdouble. The signature for
# passing data through to logP_fn are always of type '__global const cdouble'.
kernel_src = """
#define NDATA 12
cfloat logP_fn(const cfloat point[N_DIM], __global const cdouble data[NDATA]) {
    const cfloat mean = point[0];
    const cfloat log_sigma = point[1];
    if((mean < -5) || (mean > 5) || (log_sigma < -1) || (log_sigma > 0)) {
        return -INFINITY;
    }
    const cfloat sigma = exp(log_sigma);
    cdouble accumulator = 0;
    for(size_t i = 0; i < NDATA; i++) {
        accumulator += log_gaussian_normed(data[i], mean, sigma);
    }
    return accumulator;
}
"""
keys = ['x', 'y']
n_walkers = 4096
initial_state = np.random.uniform([-5, -1], [5, 0], size=(n_walkers, n_dim))
# Print the available OpenCL devices
gaps.print_devices()
blob = gaps.run_sampler(kernel_src, initial_state, keys=keys, verbose=True,
                        n_steps=2400, n_walkers=n_walkers, data=data)
plt.figure()
plt.hist2d(blob['x'].flatten(), blob['y'].flatten(), bins=64)
