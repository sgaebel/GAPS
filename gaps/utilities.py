#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assorted auxiliary functions.

These include (amongst other things) helper functions for creating
GPU buffers, compiling kernels, checking the availability of 64bit
floats, printing available devices and platform with detailed
information.

@author: Sebastian M. Gaebel
@email: sgaebel@star.sr.bham.ac.uk
"""

from .auxiliary_sources import basic_code
import numpy as np
import pyopencl as ocl


BUILD_OPTIONS = ['-cl-fp32-correctly-rounded-divide-sqrt',
                 '-Werror']


def create_read_buffer(context, host):
    """Shorthand for creating a read-only buffer on the GPU."""
    # TODO: Figure out what exactly COPY_HOST_PTR does.
    flags = ocl.mem_flags.READ_ONLY | ocl.mem_flags.COPY_HOST_PTR
    return ocl.Buffer(context, flags, hostbuf=host)


def create_write_buffer(context, host):
    """Shorthand for creating a write-only buffer on the GPU."""
    return ocl.Buffer(context, ocl.mem_flags.WRITE_ONLY, host.nbytes)


def cdouble(queue):
    """Helper which checks if 'fp64' is mentioned in the extensions of
    the device associated with the given queue, i.e. if support for
    for 64-bit floats is available."""
    if 'fp64' in queue.device.get_info(ocl.device_info.EXTENSIONS):
        return np.float64
    else:
        return np.float32


def compile_kernel(context, queue, source_code, function_name,
                   compiler_flags=None):
    """Compile the kernel given in `source_code` together with
    the GAPS math definitions and functions, and cdouble definition.
    Compiler flags can be given in addition to the default flags
    defined in `utilities.py`."""
    if cdouble(queue)(42).nbytes >= 8:
        type_definitions = """
        #define cdouble double"""
    else:
        print('WARNING: no 64bit float support available for this device.')
        type_definitions = """
        #define cdouble float"""
    flags = BUILD_OPTIONS[:]
    if compiler_flags is not None:
        flags.extend(compiler_flags)
    full_source = type_definitions + basic_code() + source_code
    program = ocl.Program(context, full_source).build(flags)
    return getattr(program, function_name)


def memory_size(n_bytes, *, SI=False, template='{:.2f} {} ({} B)'):
    """Converting a number of bytes into human readable units.

    Copied from `shed`.

    Parameters
    ----------
    n_bytes : int
        Number of bytes.
    SI : bool
        Whether to use binary units (base 1024) or SI units
        (base 1000). Keyword only argument. Default: False.
    template : string
        Template used to print the formatted memory size.
        Default: '{:.2f} {} ({} B)'.

    Returns
    -------
    value : string
        Formatted string.
    """
    if n_bytes < 0:
        raise ValueError('Memory sizes may not be negative: {!r}'
                         ''.format(n_bytes))
    if SI:
        units = ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        base = 1000
    else:
        units = ['B', 'kiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']
        base = 1024
    *units, final_unit = units

    if n_bytes < base:
        return '{:.0f} B'.format(n_bytes)
    n_units = n_bytes
    for unit in units:
        if n_units < base:
            break
        n_units /= base
    else:
        unit = final_unit
    return template.format(n_units, unit, n_bytes)


def create_context_and_queue(platform_idx=None, device_idx=None):
    """
    Convenience function to create the OpenCL context and queue needed
    to create buffers and create and call kernels.

    Parameters
    ----------
    platform_idx : int or None
    device_idx : int or None
        Indices of the chosen OpenCL platform and device on that
        platform. If both are None the device is chosen via
        `pyopencl.creat_some_context()`. Default: None.

    Returns
    -------
    context, queue : tuple
        OpenCL context and the associated command queue.

    Raises
    ------
        TODO: unavailable devices or platforms
    """
    if platform_idx is None:
        if device_idx is None:
            context = ocl.create_some_context()
            queue = ocl.CommandQueue(context)
            return context, queue
        platform_idx = 0
    if device_idx is None:
        device_idx = 0

    available_platforms = ocl.get_platforms()
    if len(available_platforms) < 1:
        raise ValueError('No platform found.')
    elif len(available_platforms) <= platform_idx:
        raise IndexError('Index {} invalid for {} available platforms.'
                         ''.format(platform_idx, len(available_platforms)))
    platform = available_platforms[platform_idx]

    available_devices = platform.get_devices()
    if len(available_devices) < 1:
        raise ValueError('No device found.')
    elif len(available_devices) <= device_idx:
        raise IndexError('Index {} invalid for {} available devices.'
                         ''.format(device_idx, len(available_devices)))
    device = available_devices[device_idx]

    context = ocl.Context([device])
    queue = ocl.CommandQueue(context)

    return context, queue


def print_devices(detail_level=0):
    """
    Print all platforms and device available, optionall with detailed
    device information.

    Parameters
    ----------
    details : int
        If >0, the function also prints device properties. Higher
        levels correspond to more (and less important) detail.
        A recommended level of detail for kernal design is 2-3.
        Maximum level is 5. Default: 0.

    Returns
    -------
    None

    Notes
    -----
    The (rough) guideline for the different levels of detail is:
     * 0 Platform and device names only
     * 1 Basic and essential
     * 2 Impacts use strongly
     * 3 Impacts use weakly
     * 4 Fine detail
     * 5 Rarely available, vendor specific, or deemed largely useless
    """
    if detail_level < 1:
        for platform_idx, platform in enumerate(ocl.get_platforms()):
            print('Platform [{}]: {} ({})'.format(platform_idx, platform.name,
                                                  platform.version))
            for device_idx, device in enumerate(platform.get_devices()):
                print('    Device [{}]: {}'.format(device_idx, device.name))
            print()  # Additional line as seperator for readability
        return


    # Specialised formatting functions for specific pieces of information.
    # Device type macros (used for ocl.device_info.TYPE):
    def device_type(info):
        """Translating the bit map into human readable categories."""
        options = {(1 << 0): 'CL_DEVICE_TYPE_DEFAULT',
                   (1 << 1): 'CL_DEVICE_TYPE_CPU',
                   (1 << 2): 'CL_DEVICE_TYPE_GPU',
                   (1 << 3): 'CL_DEVICE_TYPE_ACCELERATOR',
                   (1 << 4): 'CL_DEVICE_TYPE_CUSTOM'}
        return options.get(info, 'Undefined Device Type')


    def fp_config_formatting(info):
        """Translating the bit map into human readable categories."""
        # From: OpenCL/AMDAPPSDK-3.0/include/CL/cl.h
        options = [((1 << 0), 'CL_FP_DENORM'),
                   ((1 << 1), 'CL_FP_INF_NAN'),
                   ((1 << 2), 'CL_FP_ROUND_TO_NEAREST'),
                   ((1 << 3), 'CL_FP_ROUND_TO_ZERO'),
                   ((1 << 4), 'CL_FP_ROUND_TO_INF'),
                   ((1 << 5), 'CL_FP_FMA'),
                   ((1 << 6), 'CL_FP_SOFT_FLOAT'),
                   ((1 << 7), 'CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT')]
        # The initial line shows the bitmap, following lines
        # explicitly show the meaning and availability.
        option_breakdown = [bin(info)]
        for bitfield, option in options:
            is_available = bool(bitfield & info)
            option_breakdown.append('{}={}'.format(option, is_available))
        return ('\n\t'+' '*(device_maxwidth+3)).join(option_breakdown)


    def platform_extension_formatting(info):
        """Splitting the extensions and displaying each on aligned lines."""
        return ('\n'+' '*(platform_maxwidth+3)).join(info.split())


    def device_extension_formatting(info):
        """Splitting the extensions and displaying each on aligned lines."""
        return ('\n\t'+' '*(device_maxwidth+3)).join(info.split())

    # The following two option collections are lists of tuples with 2 or 3
    # components. The first is the detail level at which it should be
    # displayed. The second is the name of the parameter. The third is
    # optional and should, if available, be a formatting function. The
    # default is to use `str()`.

    # Complete set of possible parameters for ocl.platform_info:
    platform_info_options = [
        (1, 'NAME'),
        (4, 'PROFILE'),
        (4, 'VENDOR'),
        (1, 'VERSION'),
        (2, 'EXTENSIONS', platform_extension_formatting)]

    # Complete set of possible parameters for ocl.device_info:
    device_info_options = [
        (3, 'ADDRESS_BITS'),
        (5, 'ATTRIBUTE_ASYNC_ENGINE_COUNT_NV'),
        (1, 'AVAILABLE', bool),
        (5, 'AVAILABLE_ASYNC_QUEUES_AMD'),
        (5, 'BOARD_NAME_AMD'),
        (3, 'BUILT_IN_KERNELS'),
        (1, 'COMPILER_AVAILABLE', bool),
        (5, 'COMPUTE_CAPABILITY_MAJOR_NV'),
        (5, 'COMPUTE_CAPABILITY_MINOR_NV'),
        (5, 'CORE_TEMPERATURE_ALTERA'),
        (3, 'DOUBLE_FP_CONFIG', fp_config_formatting),
        (1, 'DRIVER_VERSION'),
        (4, 'ENDIAN_LITTLE'),
        (4, 'ERROR_CORRECTION_SUPPORT', bool),
        (4, 'EXECUTION_CAPABILITIES', bool),
        (3, 'EXTENSIONS', device_extension_formatting),
        (5, 'EXT_MEM_PADDING_IN_BYTES_QCOM'),
        (5, 'GFXIP_MAJOR_AMD'),
        (5, 'GFXIP_MINOR_AMD'),
        (5, 'GLOBAL_FREE_MEMORY_AMD'),
        (2, 'GLOBAL_MEM_CACHELINE_SIZE', memory_size),
        (2, 'GLOBAL_MEM_CACHE_SIZE', memory_size),
        (2, 'GLOBAL_MEM_CACHE_TYPE'),
        (5, 'GLOBAL_MEM_CHANNELS_AMD'),
        (5, 'GLOBAL_MEM_CHANNEL_BANKS_AMD'),
        (5, 'GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD'),
        (2, 'GLOBAL_MEM_SIZE', memory_size),
        (3, 'GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE'),
        (5, 'GPU_OVERLAP_NV'),
        (3, 'HALF_FP_CONFIG', fp_config_formatting),
        (3, 'HOST_UNIFIED_MEMORY', bool),
        (3, 'IMAGE2D_MAX_HEIGHT'),
        (3, 'IMAGE2D_MAX_WIDTH'),
        (3, 'IMAGE3D_MAX_DEPTH'),
        (3, 'IMAGE3D_MAX_HEIGHT'),
        (3, 'IMAGE3D_MAX_WIDTH'),
        (3, 'IMAGE_MAX_ARRAY_SIZE'),
        (3, 'IMAGE_MAX_BUFFER_SIZE', memory_size),
        (3, 'IMAGE_SUPPORT', bool),
        (5, 'INTEGRATED_MEMORY_NV'),
        (2, 'KERNEL_EXEC_TIMEOUT_NV'),
        (1, 'LINKER_AVAILABLE', bool),
        (5, 'LOCAL_MEM_BANKS_AMD'),
        (2, 'LOCAL_MEM_SIZE', memory_size),
        (5, 'LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD'),
        (2, 'LOCAL_MEM_TYPE'),
        (5, 'MAX_ATOMIC_COUNTERS_EXT'),
        (2, 'MAX_CLOCK_FREQUENCY'),
        (2, 'MAX_COMPUTE_UNITS'),
        (2, 'MAX_CONSTANT_ARGS'),
        (2, 'MAX_CONSTANT_BUFFER_SIZE', memory_size),
        (2, 'MAX_GLOBAL_VARIABLE_SIZE'),
        (2, 'MAX_MEM_ALLOC_SIZE', memory_size),
        (4, 'MAX_ON_DEVICE_EVENTS'),
        (4, 'MAX_ON_DEVICE_QUEUES'),
        (4, 'MAX_PARAMETER_SIZE'),
        (4, 'MAX_PIPE_ARGS'),
        (4, 'MAX_READ_IMAGE_ARGS'),
        (4, 'MAX_READ_WRITE_IMAGE_ARGS'),
        (4, 'MAX_SAMPLERS'),
        (2, 'MAX_WORK_GROUP_SIZE'),
        (2, 'MAX_WORK_ITEM_DIMENSIONS'),
        (2, 'MAX_WORK_ITEM_SIZES'),
        (3, 'MAX_WRITE_IMAGE_ARGS'),
        (4, 'MEM_BASE_ADDR_ALIGN'),
        (5, 'ME_VERSION_INTEL'),
        (4, 'MIN_DATA_TYPE_ALIGN_SIZE'),
        (1, 'NAME'),
        (4, 'NATIVE_VECTOR_WIDTH_CHAR'),
        (4, 'NATIVE_VECTOR_WIDTH_DOUBLE'),
        (4, 'NATIVE_VECTOR_WIDTH_FLOAT'),
        (4, 'NATIVE_VECTOR_WIDTH_HALF'),
        (4, 'NATIVE_VECTOR_WIDTH_INT'),
        (4, 'NATIVE_VECTOR_WIDTH_LONG'),
        (4, 'NATIVE_VECTOR_WIDTH_SHORT'),
        (5, 'NUM_SIMULTANEOUS_INTEROPS_INTEL'),
        (1, 'OPENCL_C_VERSION'),
        (5, 'PAGE_SIZE_QCOM'),
        (5, 'PARENT_DEVICE'),
        (5, 'PARTITION_AFFINITY_DOMAIN'),
        (5, 'PARTITION_MAX_SUB_DEVICES'),
        (5, 'PARTITION_PROPERTIES'),
        (5, 'PARTITION_TYPE'),
        (5, 'PCI_BUS_ID_NV'),
        (5, 'PCI_SLOT_ID_NV'),
        (5, 'PIPE_MAX_ACTIVE_RESERVATIONS'),
        (5, 'PIPE_MAX_PACKET_SIZE'),
        (4, 'PLATFORM'),
        (4, 'PREFERRED_GLOBAL_ATOMIC_ALIGNMENT'),
        (4, 'PREFERRED_INTEROP_USER_SYNC'),
        (4, 'PREFERRED_LOCAL_ATOMIC_ALIGNMENT'),
        (4, 'PREFERRED_PLATFORM_ATOMIC_ALIGNMENT'),
        (4, 'PREFERRED_VECTOR_WIDTH_CHAR'),
        (4, 'PREFERRED_VECTOR_WIDTH_DOUBLE'),
        (4, 'PREFERRED_VECTOR_WIDTH_FLOAT'),
        (4, 'PREFERRED_VECTOR_WIDTH_HALF'),
        (4, 'PREFERRED_VECTOR_WIDTH_INT'),
        (4, 'PREFERRED_VECTOR_WIDTH_LONG'),
        (4, 'PREFERRED_VECTOR_WIDTH_SHORT'),
        (4, 'PRINTF_BUFFER_SIZE'),
        (4, 'PROFILE'),
        (5, 'PROFILING_TIMER_OFFSET_AMD'),
        (3, 'PROFILING_TIMER_RESOLUTION'),
        (4, 'QUEUE_ON_DEVICE_MAX_SIZE'),
        (4, 'QUEUE_ON_DEVICE_PREFERRED_SIZE'),
        (4, 'QUEUE_ON_DEVICE_PROPERTIES'),
        (4, 'QUEUE_ON_HOST_PROPERTIES'),
        (4, 'QUEUE_PROPERTIES'),
        (4, 'REFERENCE_COUNT'),
        (5, 'REGISTERS_PER_BLOCK_NV'),
        (5, 'SIMD_INSTRUCTION_WIDTH_AMD'),
        (5, 'SIMD_PER_COMPUTE_UNIT_AMD'),
        (5, 'SIMD_WIDTH_AMD'),
        (5, 'SIMULTANEOUS_INTEROPS_INTEL'),
        (3, 'SINGLE_FP_CONFIG', fp_config_formatting),
        (5, 'SPIR_VERSIONS'),
        (5, 'SVM_CAPABILITIES'),
        (5, 'THREAD_TRACE_SUPPORTED_AMD'),
        (5, 'TOPOLOGY_AMD'),
        (1, 'TYPE', device_type),
        (1, 'VENDOR'),
        (5, 'VENDOR_ID'),
        (1, 'VERSION'),
        (5, 'WARP_SIZE_NV'),
        (5, 'WAVEFRONT_WIDTH_AMD')]

    # Options which should be displayed are selected by their assigned level.
    selector = lambda x: (x[0] <= detail_level)
    platform_options = list(filter(selector, platform_info_options))
    device_options = list(filter(selector, device_info_options))

    # Some formatting preperations
    template = '{1:<{0}} : {2}'
    global platform_maxwidth
    platform_maxwidth = max(map(len, (t[1] for t in platform_options)))
    global device_maxwidth
    device_maxwidth = max(map(len, (t[1] for t in device_options)))

    for platform_idx, platform in enumerate(ocl.get_platforms()):
        print('  Platform {}:'.format(platform_idx))
        for tup in platform_options:
            # Unpacking the option tuple. If there is no specified
            # formatting function at index 2, assume `str`.
            name = tup[1]
            formatting = str if len(tup) < 3 else tup[2]
                # Attempt to retrieve the information from the device,
                # and assume none is available if the retrieval fails.
            try:
                info = platform.get_info(getattr(ocl.platform_info, name))
            except:
                info = 'Parameter not available.'
                formatting = str
            print(template.format(platform_maxwidth, name, formatting(info)))
        for device_idx, device in enumerate(platform.get_devices()):
            print('\t  Device {}.{}:'.format(platform_idx, device_idx))
            for tup in device_options:
                # Unpacking the option tuple. If there is no specified
                # formatting function at index 2, assume `str`.
                name = tup[1]
                formatting = str if len(tup) < 3 else tup[2]
                # Attempt to retrieve the information from the device,
                # and assume none is available if the retrieval fails.
                try:
                    info = device.get_info(getattr(ocl.device_info, name))
                except:
                    info = 'Parameter not available.'
                    formatting = str
                print('\t'+template.format(device_maxwidth, name,
                                           formatting(info)))
        print()
    return


if __name__ == '__main__':
    print_devices(3)
    raise NotImplementedError
