#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAPS - A GPU Accelerated Parallel Sampler based on the affine invariant
ensemble sampler by Goodman & Weare. The pure Python equivalent is
`emcee` by Dan Foreman-Mackey and is available at http://dfm.io/emcee.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

from .convergence import (gelman_rubin, plot_gelman_rubin_brooks,
                          plot_likelihood)
from .gaps import direct_evaluation, run_sampler
from .sampler_source import ensemble_sampler_source
from .utilities import (digest_user_data, cdouble, cfloat, cshort, memory_size,
                        device_limitations, compute_group_size, print_devices,
                        create_context_and_queue)
