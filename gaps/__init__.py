#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for Python 3

@author: Sebastian Gaebel
@email: sgaebel@star.sr.bham.ac.uk
"""

# For general use
from .convergence import (gelman_rubin, plot_gelman_rubin_brooks,
                          plot_likelihood)
from .gaps import direct_call, run_sampler, EnsembleSampler
from .utilities import memory_size, print_devices, create_context_and_queue
# For testing
from .auxiliary_sources import basic_code as _basic_code
from .utilities import cdouble as _cdouble
