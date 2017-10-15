# -*- coding: utf-8 -*-
"""
Sampling a fixed 2D Gaussian.

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
x_sigma, y_sigma = 10**np.random.uniform(0, 0.5, 2)

# Define the logP function in OpenCL code.
# cfloat is the medium precision float type of GAPS.
kernel_src = """
#define X_MEAN {}
#define Y_MEAN {}
#define X_SIGMA {}
#define Y_SIGMA {}
""".format(x_mean, y_mean, x_sigma, y_sigma) + """
cfloat logP_fn(const cfloat point[N_DIM]) {
    const cfloat x = point[0];
    const cfloat y = point[1];
    if((x < -10) || (x > 10) || (y < -10) || (y > 10)) {
        return -INFINITY;
    }
    return log_gaussian(x, X_MEAN, X_SIGMA) + log_gaussian(y, Y_MEAN, Y_SIGMA);
}
"""

initial_state = np.random.uniform(-2, 2, size=(4096, 2))
blob = gaps.run_sampler(kernel_src, initial_state, keys=['x', 'y'],
                        n_steps=2400, n_walkers=4096)
plt.figure()
plt.hist2d(blob['x'].flatten(), blob['y'].flatten(), bins=64,
           range=((-10, 10), (-10, 10)))
plt.hlines([y_mean-y_sigma, y_mean, y_mean+y_sigma], -10, 10,
           linestyles=('dashed', 'solid', 'dashed'), colors='w', alpha=0.5)
plt.vlines([x_mean-x_sigma, x_mean, x_mean+x_sigma], -10, 10,
           linestyles=('dashed', 'solid', 'dashed'), colors='w', alpha=0.5)
plt.show()
