#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for Python 3

@author: Sebastian Gaebel
@email: sgaebel@star.sr.bham.ac.uk
"""

from .auxiliary_sources import basic_code
from .utilities import create_context_and_queue
import numpy as np
import pyopencl as ocl

READ_ONLY = ocl.mem_flags.READ_ONLY
WRITE_ONLY = ocl.mem_flags.WRITE_ONLY
COPY_HOST_PTR = ocl.mem_flags.COPY_HOST_PTR


class EnsembleSampler:
    def __init__():
        raise NotImplementedError

    def reset():
        raise NotImplementedError

    def sample():
        raise NotImplementedError

    def save_to_disk():
        raise NotImplementedError

    def load_from_disk():
        raise NotImplementedError


def run_sampler():
    raise NotImplementedError


def direct_call(parameters, function_source, user_data=None,
                platform_idx=None, device_idx=None, function_name='logL_fn'):
    """
    Directly call the log-likelihood independently of the
    sampler.

    Parameters
    ----------
    parameters : convertable to a 2D float array
        Points for which the function is to be evaluated. This needs
        to be convertable to a 2D float array with shape (N_EVAL, N_DIM),
        where N_EVAL is the number of points for which the function
        will be evaluated, and N_DIM is the number of dimensions of
        the parameter space.
    function_source : string
        Source code for the function to be called. The call signature
        needs to be of the form `function_name(__global double parameter_vector, user_data[???])`
    user_data : numpy array or list of arrays, or None
        If given, this is either a single numpy array, or a list of
        numpy arrays. These are passed to `function_name` after the
        parameter vector in order. Array shapes are available as
        constants named `USER_DIM_Y` `USER_DIM_X_Y` where `X` is the
        position in the list (if given) and `Y` is the index of the
        dimension for the given array. Default: None.
        NOTE: Currently all user data is converted to np.float64 and
        passed along to `function_name` as double.
    platform_idx : int or None
    device_idx : int or None
        Indices of the chosen OpenCL platform and device. These
        indices are passed through to `create_context_and_queue`,
        refer to its documentation for more details.
        Default: None.
    function_name : string
        Name of the user-defined function to be evaluated.
        Default: 'logL_fn'.

    Returns
    -------
    value : numpy array of shape (N_EVAL,)
    """
    context, queue = create_context_and_queue(platform_idx=platform_idx,
                                              device_idx=device_idx)
    parameters_cpu = np.array(parameters, dtype=np.float64)
    parameters_gpu = ocl.Buffer(context, READ_ONLY | COPY_HOST_PTR,
                                hostbuf=parameters_cpu)
    n_eval, n_dim = parameters_cpu.shape
    logL_values_cpu = np.zeros(n_eval, dtype=np.float64)
    logL_values_gpu = ocl.Buffer(context, WRITE_ONLY, logL_values_cpu.nbytes)

    defines = """
    #define N_DIM {n_dim}
    #define N_EVAL {n_dim}
    """.format(n_dim=n_dim, n_eval=n_eval)
    if user_data is None:
        user_data_arguments = ''
        user_data_call = ''
        user_data_gpu = []
    elif isinstance(user_data, np.ndarray):
        user_data = user_data.astype(np.float64)
        for idx, length in enumerate(user_data.shape):
            defines += '\n#define USER_DIM_{} {}'.format(idx, length)
        user_data_arguments = ', __global const double user_data[{}]'.format(
            ', '.join(map(str, user_data.shape)))
        user_data_call = ', user_data'
        user_data_gpu = [ocl.Buffer(context, READ_ONLY | COPY_HOST_PTR,
                                    hostbuf=user_data)]
    else:
        for arr_idx, array in enumerate(user_data):
            user_data[arr_idx] = array.astype(np.float64)
        user_data_arguments = ''
        user_data_call = ''
        user_data_gpu = []
        for arr_idx, array in enumerate(user_data):
            for idx, length in enumerate(user_data.shape):
                defines += '\n#define USER_DIM_{}_{} {}'.format(arr_idx, idx, length)
            user_data_arguments += ', __global const double user_data_{}[{}]'.format(
                arr_idx, ', '.join(map(str, user_data.shape)))
            user_data_call += ', user_data_{}'.format(arr_idx)
            user_data_gpu.append(ocl.Buffer(context, READ_ONLY | COPY_HOST_PTR,
                                            hostbuf=user_data[arr_idx]))
    kernel_source = """
    ___kernel void DirectEvaluation(__global double parameters[N_EVAL][N_DIM],
                                    __global double logL_values[N_EVAL] USER_DATA_ARGUMENTS) {
        const size_t eval_idx = get_global_idx(0);
        double parameter_vector[N_DIM];
        for(size_t i = 0; i < N_DIM; i++) {
            parameter_vector[i] = parameters[eval_idx][i];
        }
        logL_values[eval_idx] = GAPS_FUNC_NAME(parameter_vector USER_DATA_CALL);
        return;
    }
    """
    kernel_source = kernel_source.replace('GAPS_FUNC_NAME', function_name)
    kernel_source = kernel_source.replace('USER_DATA_ARGUMENTS', user_data_arguments)
    kernel_source = kernel_source.replace('USER_DATA_CALL', user_data_call)
    source_code = defines + basic_code(queue) + function_source + kernel_source

    program = ocl.Program(context, source_code).build()
    event = program.DirectEvaluation(queue, (1,), (n_eval,),
                                     parameters_gpu, logL_values_gpu,
                                     *user_data_gpu)
    ocl.enqueue_barrier(queue, wait_for=[event])
    ocl.enqueue_copy(queue, logL_values_cpu, logL_values_gpu)
    return logL_values_cpu


if __name__ == '__main__':
    pass
