#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for Python 3

@author: Sebastian Gaebel
@email: sgaebel@star.sr.bham.ac.uk
"""

from .auxiliary_sources import basic_code
from .utilities import (create_context_and_queue, cdouble, create_read_buffer,
                        create_write_buffer, compile_kernel, device_limitations)
import numpy as np
import pyopencl as ocl


def direct_evaluation(source_code, platform_idx=None, device_idx=None,
                      read_only_arrays=[], write_only_shapes=[],
                      kernel_name='test_kernel'):
    """This function provides a convenient interface to directly call
    custom functions through PyOpenCL. The intended use for this is
    code testing and prototyping, as well as for general unit tests.

    Parameters
    ----------
    source_code : string
        String containing the user defined kernel and functions.
    platform_idx : integer or None
    device_idx : integer or None
        Integers specifying the OpenCL platform and device to be used
        for execution. If left as None, the device is selected
        automatically (but in no way optimised). Default: None.
    read_only_arrays : list of arrays
    write_only_shapes : list of tuples
        The read only arrays are converted internally to the GAPS
        cdouble type. The buffers created for each read or write onyl
        array are then passed in order (read, then write) to the
        kernel. Default: [].
    kernel_name : string
        Name of the kernel. Default: 'test_kernel'.

    Returns
    -------
    write_only_cpu : list of arrays
        The numpy arrays into which the write_only arrays were copied
        after the kernel execution. Shapes are those given via
        `write_only_shapes`.
    """
    context, queue = create_context_and_queue(platform_idx, device_idx)
    dtype = cdouble(queue)
    read_only_cpu = [np.array(x, dtype=dtype) for x in read_only_arrays]
    write_only_cpu = [np.empty(x, dtype=dtype) for x in write_only_shapes]
    read_only_gpu = [create_read_buffer(context, x) for x in read_only_cpu]
    write_only_gpu = [create_write_buffer(context, x) for x in write_only_cpu]
    buffers = read_only_gpu + write_only_gpu
    kernel = compile_kernel(context, queue, source_code, kernel_name)
    event = kernel(queue, (1,), (1,), *buffers)
    ocl.enqueue_barrier(queue, wait_for=[event])
    for idx, write_buffer in enumerate(write_only_gpu):
        ocl.enqueue_copy(queue, write_only_cpu[idx], write_buffer)
    return write_only_cpu


def run_sampler(logP_fn_source, data=None, state=None, logP_state=None,
                prior_bounds=None, n_steps=600, n_walkers=1024, b_burnin=120,
                group_size=None, platform_idx=None, device_idx=None,
                flatten=True, keys=None, debug_output=False):
    """Run the OpenCL based ensemble sampler.

    Parameters
    ----------
    logP_fn_source : str
        Source code of the logP function from which samples are to be
        drawn. The function signature is required to be of the form
        `cfloat logP_fn(const cfloat point, $USER$)` where $USER$ should  # TODO: const?
        accept the user_data passed to the sampler in order (if any).
    data : float array-like
        User defined data objects which are immaterial to the sampler,
        but passed through to `logP_fn`. Default: None.
    state : numpy array
        The initial state of the sampler. This needs to be convertable
        to a floating point numpy array of shape (n_walkers, n_dim).
        Default: None.
    logP_state : numpy array
        If `state` is given, the corresponding logP_values may be made
        available as well. These are assumed to be accurate without
        further checks. This may be useful if the sampling process
        is to be split up into smaller chunks as it avoids duplicated
        computations. The shape of this array must be (n_walkers,).
        Default: None.
    prior_bounds : list of float pairs
        Instead of passing a initial state, the prior bounds may be
        given instead. In this case the initial state is drawn from
        a uniform distribution between the prior bounds. Passing both
        `state` and `prior_bounds` will raise ValueError. Default: None.
    n_steps : int > 0
        Number of steps to be taken by each walker. Default: 600.
    n_walkers : int > 0
        Number of walkers in the ensemble. Default: 1024.
    n_burnin : int >= 0
        Number of discarded initial steps to be taken by each walker.
        If this value is non-zero the total number of steps taken by
        each walker will be n_steps+n_burnin. Default: 120.
    group_size : int
        Based on architectural and device specific limitations the
        ensemble may be split up into smaller subgroups. The size of
        these groups is chosen automatically is `group_size` is None.
        If specified, the number of walker must be a multiple of this
        group size. Larger group sizes are generally preferred.
        For large numbers of walkers and small group sizes device
        limitations on the number of work items may become relevant.
        Default: None.
    platform_idx : int
    device_idx : int
        Indices of the chosen OpenCL platform and device on that
        platform. If both are None the device is chosen via
        `pyopencl.creat_some_context()`. Default: None.
    flatten : bool
        If True, the returns chains are flattened to hide the split
        by walker. Default: True.
    keys : list
        If given, this list will be used to generate dict-like
        structured numpy array, which is returned instead of the usual
        tuple.
    debug_output : bool
        The sampler may produce additional output which is useful
        for debugging logp_fn. This includes proposed points
        irregardless of whether they were accepted or not, as well as
        the logP values for those points. Optionally an additional
        debug_fn may be passed via `debug_fn_source` (which has the
        same requirements and function signature as `logP_fn`) whose
        output will be returned as well. Default: False.
    debug_fn_source : str
        Source code defining `debug_fn`, which has the same requirements
        and function signature as `logP_fn`. Passing a values to this
        argument automatically enables `debug_output` as well.
        Default: None.

    Returns
    -------
    Posterior chains, logP values, and optional debug output. If `keys`
    is given, chains and logP_values will be combined into a single
    dict-like structured array. If flatten is True, the walker and step
    axes are combined into a single axis in all data products.
    If debug_output is True, the return values will contain an
    additional dict of data useful for debugging.

    Notes
    -----
    A more detailed description on the requirements of logP_fn and
    additional useful information is given in `README.md`.

    Examples of output structure:
        flatten==True, keys==None:
            chains.shape == (n_walkers*n_steps, n_dim)
            logP_values.shape == (n_walkers*n_steps)
        flatten==False, keys==None:
            chains.shape == (n_walkers, n_steps, n_dim)
            logP_values.shape == (n_walkers, n_steps)
        flatten==True, keys!=None:
            return_value.shape == (n_walkers*n_steps)
        flatten==False, keys!=None:
            return_value.shape == (n_walkers, n_steps)
    """
    raise NotImplementedError



if __name__ == '__main__':
    pass
