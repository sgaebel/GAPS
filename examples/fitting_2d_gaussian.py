# -*- coding: utf-8 -*-
"""
Fitting a 2D Gaussian.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

import matplotlib.pyplot as plt
import numpy as np
import gaps


# Print the available platform and devices
gaps.print_devices()

# Define a random 2D gaussian
x_mean, y_mean = np.random.uniform(-5, 5, 2)
x_sigma, y_sigma = 10**np.random.uniform(-1, 0.5, 2)

# Generate some samples
samples = np.random.normal((x_mean, y_mean), (x_sigma, y_sigma), size=(240, 2))

# Define the logP function in OpenCL code.
# cfloat is the medium precision float type of GAPS.
kernel_src = """
#define N_POINTS {}
""".format(samples.shape[0]) + """
cfloat logP_fn(const cfloat point[N_DIM],
               __global const cdouble samples[N_POINTS][2]) {
    const cfloat x_mean = point[0];
    const cfloat y_mean = point[1];
    const cfloat x_log_sigma = point[2];
    const cfloat y_log_sigma = point[3];
    if((x_mean < -10) || (x_mean > 10) ||
       (y_mean < -10) || (y_mean > 10) ||
       (x_log_sigma < -3) || (x_log_sigma > 3) ||
       (y_log_sigma < -3) || (y_log_sigma > 3)) {
        return -INFINITY;
    }
    cdouble accumulator = 0;
    for(size_t i = 0; i < N_POINTS; i++) {
        accumulator += log_gaussian(samples[i][0], x_mean, exp(x_log_sigma));
        accumulator += log_gaussian(samples[i][1], y_mean, exp(y_log_sigma));
    }
    return accumulator;
}
"""

initial_state = np.random.uniform((-10, -10, -3, -3), (10, 10, 3, 3), size=(4096, 4))
blob = gaps.run_sampler(kernel_src, initial_state, n_steps=2400, n_walkers=4096,
                        keys=['x_mean', 'y_mean', 'x_log_sigma', 'y_log_sigma'],
                        data=samples)
opts = dict(linestyles=('dashed', 'solid', 'dashed'), colors='r', alpha=0.75)

plt.figure()
H = plt.hist(blob['x_mean'].flatten(), bins=120)
plt.vlines(x_mean, 0, np.max(H[0]), **opts)
plt.xlabel('x_mean')
plt.show()

plt.figure()
H = plt.hist(blob['y_mean'].flatten(), bins=120)
plt.vlines(y_mean, 0, np.max(H[0]), **opts)
plt.xlabel('y_mean')
plt.show()

plt.figure()
H = plt.hist(blob['x_log_sigma'].flatten(), bins=120)
plt.vlines(np.log(x_sigma), 0, np.max(H[0]), **opts)
plt.xlabel('x_log_sigma')
plt.show()

plt.figure()
H = plt.hist(blob['y_log_sigma'].flatten(), bins=120)
plt.vlines(np.log(y_sigma), 0, np.max(H[0]), **opts)
plt.xlabel('y_log_sigma')
plt.show()
