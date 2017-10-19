#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core functions required to run the sampler, or evaluate an OpenCL
function directly.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""

from .utilities import (create_context_and_queue, cdouble, create_read_buffer,
                        create_write_buffer, compile_kernel, memory_size,
                        digest_user_data, device_limitations,
                        compute_group_size)
from .sampler_source import ensemble_sampler_source
import numpy as np
import pyopencl as ocl


def direct_evaluation(source_code, platform_idx=None, device_idx=None,
                      read_only_arrays=[], write_only_shapes=[],
                      kernel_name='test_kernel',
                      global_size=(1,), local_size=(1,)):
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
    global_size : tuple of ints
    local_size : tuple of ints
        Number of global or local work items to use for computation.
        The default is (1,) which executes a single function call.
        For other sizes one must use `get_global_id(0)` and/or
        `get_local_id(0)` for code to be  aware of their role in the
        parallel execution.

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
    event = kernel(queue, global_size, local_size, *buffers)
    ocl.enqueue_barrier(queue, wait_for=[event])
    for idx, write_buffer in enumerate(write_only_gpu):
        ocl.enqueue_copy(queue, write_only_cpu[idx], write_buffer)
    return write_only_cpu


def run_sampler(logP_fn_source, state, logP_state=None, data=None,
                n_dim=None, n_steps=600, n_walkers=1024,
                group_size=None, scale_parameter=2., platform_idx=None,
                device_idx=None, keys=None, debug_output=False, verbose=False):
    """Run the OpenCL based ensemble sampler.

    Parameters
    ----------
    logP_fn_source : str
        Source code of the logP function from which samples are to be
        drawn. The function signature is required to be of the form
        `cfloat logP_fn(const cfloat point, $USER$)` where $USER$ should
        accept the user_data passed to the sampler in order (if any).
    state : numpy array
        The initial state of the sampler. This needs to be a numpy
        array of shape (n_walkers, n_dim).
    data : Numpy array or list of numpy arrays.
        User defined data objects which are immaterial to the sampler,
        but passed through to `logP_fn`. Note that objects interacting
        with OpenCL need to be numpy arrays, and are converted to
        `cdouble` automatically.
        Default: None.
    logP_state : numpy array
        If `state` is given, the corresponding logP_values may be made
        available as well. These are assumed to be accurate without
        further checks. This may be useful if the sampling process
        is to be split up into smaller chunks as it avoids duplicated
        computations. The shape of this array must be (n_walkers,).
        Default: None.
    n_dim : int
        Dimensionality of the parameter space. This value is inferred
        from the required argument `state`, but passing it directly
        allows for an additional check. Default: None.
    n_steps : int > 0
        Number of steps to be taken by each walker. Default: 600.
    n_walkers : int > 0
        Number of walkers in the ensemble. Default: 1024.
    group_size : int
        Based on architectural and device specific limitations the
        ensemble may be split up into smaller subgroups. The size of
        these groups is chosen automatically is `group_size` is None.
        If specified, the number of walker must be a multiple of this
        group size. Larger group sizes are generally preferred.
        For large numbers of walkers and small group sizes device
        limitations on the number of work items may become relevant.
        Default: None.
    scale_parameter : float
        The proposal scale parameter. Default: 2.
    platform_idx : int
    device_idx : int
        Indices of the chosen OpenCL platform and device on that
        platform. If both are None the device is chosen via
        `pyopencl.creat_some_context()`. Default: None.
    keys : list
        If given, this list will be used to generate dict-like
        structured numpy array, which is returned instead of the usual
        tuple.
    debug_output : bool
        The sampler may produce additional output which is useful
        for debugging logp_fn. This includes proposed points
        irregardless of whether they were accepted or not, as well as
        the logP values for those points. Optionally an additional
        `debug_fn` may be passed within `logP_fn_source` (which has the
        same requirements and function signature as `logP_fn`) whose
        output will be returned seperately. Default: False.
    verbose : bool
        Toggles verbose output, which includes some less impactful
        warnings as well as progress and timing information.
        Default: False

    Returns
    -------
    state_dict : dict
        Dictionary containing data products, e.g. `state_blob.flatchain`.
        If keys are given, the tuple is populated with unflattened
        chains under `state_blub.key`. Debug values are contained in the
        same way.
        Guarateed attributes:
            chain, flatchain, logP, flatlogP, final_state, final_logP
        With debug_output == True:
            debug_values, proposed_points, acceptance_counter

    Notes
    -----
    A more detailed description on the requirements of logP_fn and
    additional useful information is given in `README.md`.

    Array formatting:
    state_blob['chains'].shape == (n_walkers, n_steps, n_dim)
    state_blob['flatchain'].shape == (n_walkers*n_steps, n_dim)
    state_blob['logP'].shape == (n_walkers, n_steps, n_dim)
    state_blob['flatlogP'].shape == (n_walkers*n_steps, n_dim)
    state_blob['final_state'].shape == (n_walkers, n_dim)
    state_blob['final_logP'].shape == (n_walkers,)
    state_blob['debug_values'].shape == (n_walkers, n_steps)
    state_blob['proposed_points'].shape == (n_walkers, n_steps, n_dim)
    state_blob['acceptance_counter'].shape == (n_walkers,)
    state_blob[key].shape == (n_walkers, n_steps)
    """
    compiler_flags = []

    # The datatype may be required for conversion, so we create context
    # and queue before some of the checks
    context, queue = create_context_and_queue(platform_idx, device_idx)
    cdouble_t = cdouble(queue)
    cfloat_t = np.float32
    # Convert user data to numpy and generate OpenCL code to pass user data
    user_data_cpu, func_name, func_def = digest_user_data(data, cdouble_t)
    # Check that the shape of `state` is valid.
    if len(state.shape) != 2:
        raise ValueError('The given state must have shape (n_walkers, n_dim). '
                         'Given: {}.'.format(state.shape))
    if state.shape[0] != n_walkers:
        raise ValueError('Mismatch between n_walkers ({}) and state.shape[0] '
                         '({}).'.format(n_walkers, state.shape[0]))
    # Cross-check n_dim with other possible sources.
    if n_dim is None:
        n_dim = state.shape[1]
    elif n_dim != state.shape[1]:
        raise ValueError('Mismatch between n_dim ({}) and state.shape[1] ({}).'
                         ''.format(n_dim, state.shape[1]))
    if keys is not None and len(keys) != n_dim:
        raise ValueError('Mismatch between n_dim ({}) and len(keys) ({}).'
                         ''.format(n_dim, len(keys)))

    # Check for negative values
    if any(x < 0 for x in (n_walkers, n_steps, n_dim)):
        raise ValueError('All of (n_walkers, n_steps, n_dim) must be'
                         ' non-negative: {}.'
                         ''.format((n_walkers, n_steps, n_dim)))
    # Check if a logP_state exists and has valid shape
    if logP_state is not None:
        if logP_state.shape != (n_walkers,):
            raise ValueError('Mismatch between n_walkers ({}) and '
                             'logP_state.shape[0] ([}).'
                             ''.format(n_walkers, logP_state.shape[0]))
        # The existing logP values are passed along with the user data, though
        # it is not passed along further to logP_fn calls.
        compiler_flags.append('-DINITIAL_LOGP')
        user_data_cpu.append(logP_state.astype(cdouble_t))
        func_def += ', __global const cfloat given_logP[N_WALKERS]'
    # Check for debug request, and that the required function definition
    # is present if that is the case.
    if debug_output and 'cdouble debug_fn(' not in logP_fn_source:
        raise ValueError('Debug output requires `debug_fn` to be defined.')
    if debug_output:
        compiler_flags.append('-DDEBUG_OUTPUT')

    # Additional conversion and set up on the CPU side.
    initial_state_cpu = state.astype(cfloat_t)
    uniform_random_cpu = np.random.uniform(size=(n_walkers, n_steps, 3)).astype(cfloat_t)
    samples_cpu = np.zeros((n_walkers, n_steps, n_dim), dtype=cfloat_t)
    logP_values_cpu = np.zeros((n_walkers, n_steps), dtype=cfloat_t)
    acceptance_counter_cpu = np.zeros(n_walkers, dtype=np.uint32)
    if debug_output:
        debug_values_cpu = np.zeros((n_walkers, n_steps), dtype=cfloat_t)
        proposed_points_cpu = np.zeros((n_walkers, n_steps, n_dim), dtype=cfloat_t)

    # Create GPU based buffers for all data.
    user_data_gpu = [create_read_buffer(context, x) for x in user_data_cpu]
    initial_state_gpu = create_read_buffer(context, initial_state_cpu)
    uniform_random_gpu = create_read_buffer(context, uniform_random_cpu)
    samples_gpu = create_write_buffer(context, samples_cpu)
    logP_values_gpu = create_write_buffer(context, logP_values_cpu)
    acceptance_counter_gpu = create_write_buffer(context, acceptance_counter_cpu)
    if debug_output:
        debug_values_gpu = create_write_buffer(context, debug_values_cpu)
        proposed_points_gpu = create_write_buffer(context, proposed_points_cpu)

    # Check memory requirements and compute/check group size
    limitations = device_limitations(queue.device)
    global_avail = limitations['global_memory']
    local_avail = limitations['local_memory']
    computed_group_size = compute_group_size(queue.device, n_dim, cfloat_t)
    if group_size is None:
        group_size = computed_group_size
        if verbose:
            print('Selected group size: {}'.format(group_size))
    elif group_size > limitations['group_size']:
        group_size = limitations['group_size']
        print('WARNING: Given group size ({}) is above the device limit ({}) '
              'and has been overwritten to match the limit.'
              ''.format(group_size, limitations['group_size']))
    if n_walkers < group_size:
        if verbose:
            print('Number of walkers ({}) below the estimated optimum '
                  'group size ({}).'.format(n_walkers, group_size))
        group_size = n_walkers
    if computed_group_size < group_size:
        print('WARNING: Estimated ideal group size ({}) is smaller then the '
              'given group size ({}).'.format(memory_size(computed_group_size),
                                              memory_size(group_size)))
    global_size = sum(map(lambda x: x.nbytes,
                          user_data_cpu + [initial_state_cpu, samples_cpu,
                                           uniform_random_cpu, logP_values_cpu,
                                           acceptance_counter_cpu]))
    if debug_output:
        global_size += debug_values_cpu.nbytes + proposed_points_cpu.nbytes
    # Breakdown: 2 buffers, 1 number per dimension + 1 for the logP value,
    #            each with cfloat_t.nbytes
    local_size = 2 * group_size * (n_dim+1) * cfloat_t(1.0).nbytes
    # 0.9 is a safety factor since OpenCL may allocate some additional space
    if global_size > 0.9 * global_avail:
        raise MemoryError('Insufficient global memory: {} used out of {} '
                          'required.'.format(memory_size(global_size),
                                             memory_size(global_avail)))
    if local_size > 0.9 * local_avail:
        raise MemoryError('Insufficient local memory: {} used out of {} '
                          'required.'.format(memory_size(local_size),
                                             memory_size(local_avail)))

    # Build the argument list for invocing the sampler kernel.
    kernel_arguments = [initial_state_gpu, uniform_random_gpu, samples_gpu,
                        logP_values_gpu, acceptance_counter_gpu]
    kernel_arguments += user_data_gpu
    if debug_output:
        kernel_arguments += [debug_values_gpu, proposed_points_gpu]

    # Combine sources and build compiler flags
    full_source = ensemble_sampler_source.replace('GAPS_FUNC_DEF', func_def)
    full_source = full_source.replace('GAPS_FUNC_CALL', func_name)
    full_source = logP_fn_source + full_source
    compiler_flags += ['-DN_WALKERS={}'.format(n_walkers),
                       '-DN_SAMPLES={}'.format(n_steps),
                       '-DN_DIM={}'.format(n_dim),
                       '-DSCALE_PARAMETER={}'.format(scale_parameter),
                       '-DGROUP_SIZE={}'.format(group_size)]

    # Build the kernel and invoke it
    kernel = compile_kernel(context, queue, full_source, 'ensemble_sampling',
                            compiler_flags)
    event = kernel(queue, (n_walkers,), (group_size,), *kernel_arguments)
    # Enqueue a barrier to wait for the kernel to complete, then copy
    # the results back to the CPU environment.
    ocl.enqueue_barrier(queue, wait_for=[event])
    ocl.enqueue_copy(queue, samples_cpu, samples_gpu)
    ocl.enqueue_copy(queue, logP_values_cpu, logP_values_gpu)
    ocl.enqueue_copy(queue, acceptance_counter_cpu, acceptance_counter_gpu)
    if debug_output:
        ocl.enqueue_copy(queue, debug_values_cpu, debug_values_gpu)
        ocl.enqueue_copy(queue, proposed_points_cpu, proposed_points_gpu)

    # Calculate the acceptance ratio and display it if verbose.
    if verbose:
        acceptance_ratio = acceptance_counter_cpu.astype(np.float) / n_steps
        print('Acceptance Ratio: {:.2%} \u00B1 {:.2%}'
              ''.format(np.mean(acceptance_ratio), np.std(acceptance_ratio)))

    # Build an object containing all relevant data products
    state_dict = {}
    state_dict['chain'] = samples_cpu
    state_dict['flatchain'] = samples_cpu.reshape(n_walkers*n_steps, n_dim)
    state_dict['logP'] = logP_values_cpu
    state_dict['flatlogP'] = logP_values_cpu.reshape(n_walkers*n_steps)
    state_dict['final_state'] = samples_cpu[:, -1, :]
    state_dict['final_logP'] = logP_values_cpu[:, -1]
    if keys is not None:
        for idx, key in enumerate(keys):
            state_dict[key] = samples_cpu[..., idx]
    if debug_output:
        state_dict['debug_values'] = debug_values_cpu
        state_dict['proposed_points'] = proposed_points_cpu
        state_dict['acceptance_counter'] = acceptance_counter_cpu
    return state_dict


if __name__ == '__main__':
    pass
