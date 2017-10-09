# -*- coding: utf-8 -*-
"""
Examples to illustrate the basic use of GAPS.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

from emcee import EnsembleSampler
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# There is no installation functionality yet. To use the module, we
# add the path manually for now.
import sys
sys.path.append(r'/path/to/GAPS')
import gaps


# For comparison, define some distribution in Python.
def logP(point):
    x, y = point
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    radial = norm.logpdf(r, loc=1, scale=0.1)
    angular = np.log(np.cos(2.7**np.abs(phi) - 1)+1)
    return radial + angular


x_axis = np.linspace(-2, 2, 720)
y_axis = np.linspace(-2, 2, 720)
x, y = np.meshgrid(x_axis, y_axis, indexing='ij')
r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
z = np.exp(logP((x, y)))
plt.figure()
plt.pcolormesh(x, y, z)


# As an additional comparison, use emcee to sample the distribution.
n_walkers, n_steps, n_dim = 120, 600, 2
initial_state = np.random.uniform(-2, 2, size=(n_walkers, n_dim))
sampler = EnsembleSampler(n_walkers, n_dim, logP)
sampler.run_mcmc(initial_state, n_steps)
plt.figure()
plt.hist2d(sampler.flatchain[:, 0], sampler.flatchain[:, 1], bins=64)


# Define the logP function in OpenCL code.
# cfloat is the medium precision float type of GAPS.
kernel_src = """
cfloat logP_fn(const cfloat point[N_DIM]) {
    const cfloat x = point[0];
    const cfloat y = point[1];
    const cfloat r = sqrt(x*x + y*y);
    const cfloat phi = atan2(y, x);
    const cfloat radial = log_gaussian_normed(r, 1., 0.1);
    const cfloat angular = log(cos(powr(2.7f, fabs(phi)) - 1) + 1.);
    return radial + angular;
}
"""
keys = ['x', 'y']
initial_state = np.random.uniform(-2, 2, size=(4096, 2))
# Print the available OpenCL devices
gaps.print_devices()
blob = gaps.run_sampler(kernel_src, initial_state, keys=keys, verbose=True,
                        n_steps=2400, n_walkers=4096)
plt.figure()
plt.hist2d(blob['x'].flatten(), blob['y'].flatten(), bins=64)
