#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for `gaps/auxiliary_sources.py`.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""


if __name__ == '__main__':
    import os
    os.chdir('..')

import pytest
import gaps
import numpy as np
import pyopencl as ocl
import scipy.integrate
import scipy.misc
import scipy.stats


VISUAL = None
VERBOSE = None
N_MATH_TESTS = 3


device_list = []
for platform_idx, platform in enumerate(ocl.get_platforms()):
    for device_idx, device in enumerate(platform.get_devices()):
        device_list.append((platform_idx, platform, device_idx, device))


def close(x, y, tolerance):
    return np.isclose(x, y, rtol=tolerance, atol=tolerance)


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)
    return


def type_and_tolerance(platform_idx, device_idx):
    device = ocl.get_platforms()[platform_idx].get_devices()[device_idx]
    if 'fp64' in device.get_info(ocl.device_info.EXTENSIONS):
        return np.float64, 1e-8
    else:
        return np.float32, 1e-4


# Run tests for all available platforms and devices
@pytest.fixture(params=device_list)
def args(request):
    return request.param


# %% Math functions

def test_math_function_sum(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = sum(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.sum(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Sum [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_product(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = product(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.product(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Product [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_logsumexp(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = logsumexp(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = scipy.misc.logsumexp(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('LogSumExp [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_logaddexp(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[2],
                              __global cdouble ret_value[1]) {
        ret_value[0] = logaddexp(values[0], values[1]);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 2).astype(cdouble)
        y_expected = np.logaddexp(x[0], x[1])
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('LogAddExp [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_mean(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = mean(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.mean(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Mean [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_stddev(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = stddev(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.std(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('StdDev [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_iter_min(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = iter_min(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.min(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Min [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_math_function_iter_max(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = iter_max(values, N);
        return;
    }
    """

    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Max [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


# %% Distributions


def test_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[3],
                              __global cdouble ret_value[1]) {
        ret_value[0] = gaussian(values[2], values[0], values[1]);
        return;
    }
    """

    # Expected values: scipy.stats.norm.pdf(x, mean, sigma)
    test_cases = [(42, 10.1, 31.2, 0.0222996875223),
                  (-12.4, 35, 5.6, 0.00998639619492),
                  (0, 1, 0, 0.398942280401),
                  (0, 1, 1, 0.241970724519),
                  (-2, 0.2, 1, 2.76535477492e-49),
                  (-2, 0.2, -12, 0.0)]
    for idx, (mean, sigma, x, y_expected) in enumerate(test_cases):
        values = np.array([mean, sigma, x], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Gaussian [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[3],
                              __global cdouble ret_value[1]) {
        ret_value[0] = log_gaussian(values[2], values[0], values[1]);
        return;
    }
    """

    # Expected values: scipy.stats.norm.logpdf(x, mean, sigma)
    test_cases = [(42, 10.1, 31.2, -3.80318261307),
                  (-12.4, 35, 5.6, -4.60653149265),
                  (0, 1, 0, -0.918938533205),
                  (0, 1, 1, -1.4189385332),
                  (-2, 0.2, 1, -111.809500621),
                  (-2, 0.2, -12, -1249.30950062)]
    for idx, (mean, sigma, x, y_expected) in enumerate(test_cases):
        values = np.array([mean, sigma, x], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('LogGaussian [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_trunc_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[5],
                              __global cdouble ret_value[1]) {
        ret_value[0] = trunc_gaussian(values[0], values[1], values[2],
                                      values[3], values[4]);
        return;
    }
    """

    # scipy.stats.truncnorm.pdf(x, a, b, mean, sigma)
    # where: a, b = (low - mean) / sigma, (high - mean) / sigma
    test_cases = [(3.2, 0.9, 0, 2, 3.2, 0.0),
                  (3.2, 0.9, 0, 2, 0, 0.00875685427076),
                  (3.2, 0.9, 0, 2, 2, 2.00206715722),
                  (3.2, 0.9, 0, 2, -0.1, 0.0),
                  (-21, 2, -13, 4, -1.7, 3.78373778724e-17),
                  (13, 46, -23, 64, 52.6, 0.00922116843863)]
    for idx, (mean, sigma, low, high, x, y_expected) in enumerate(test_cases):
        values = np.array([x, mean, sigma, low, high], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('TruncGaussian [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_trunc_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[5],
                              __global cdouble ret_value[1]) {
        ret_value[0] = log_trunc_gaussian(values[0], values[1], values[2],
                                          values[3], values[4]);
        return;
    }
    """

    # scipy.stats.truncnorm.logpdf(x, a, b, mean, sigma)
    # where: a, b = (low - mean) / sigma, (high - mean) / sigma
    test_cases = [(3.2, 0.9, 0, 2, 3.2, -np.inf),
                  (3.2, 0.9, 0, 2, 0, -4.73791854004),
                  (3.2, 0.9, 0, 2, 2, 0.694180225394),
                  (3.2, 0.9, 0, 2, -0.1, -np.inf),
                  (-21, 2, -13, 4, -1.7, -37.8132342272),
                  (13, 46, -23, 64, 52.6, -4.68625352074)]
    for idx, (mean, sigma, low, high, x, y_expected) in enumerate(test_cases):
        values = np.array([x, mean, sigma, low, high], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        if np.isinf(y_expected):
            vprint('LogTruncGaussian [{:>2}]: {:>13.6e} vs. {:>13.6e}'
                   ''.format(idx, y, y_expected))
            assert np.isinf(y)
            assert y < 0
            continue
        vprint('LogTruncGaussian [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_power_law(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[4],
                              __global cdouble ret_value[1]) {
        ret_value[0] = power_law(values[0], values[1], values[2], values[3]);
        return;
    }
    """

    # If slope == -1: 1/(value * (np.log(high) - np.log(low)))
    # else: value**slope * (slope+1) / (high**(slope+1) - low**(slope+1))
    test_cases = [(-3, 6.6e-9, 5.1e0, 4.4e-3, 1.0227272727272731e-09),
                  (-2, 9.4e-2, 6.4e1, 9.2e0, 0.001112219583855943),
                  (-1, 5.8e-8, 3.8e2, 5.4e1, 0.00081929493294983645),
                  (0, 1.5e-8, 8.4e1, 5.2e-4, 0.011904761906887752),
                  (1, 1.2e-5, 6.1e-3, 7.9e-4, 42.461868167401661),
                  (2, 4.7e-2, 5.7e1, 2.4e1, 0.0093308062452077192),
                  (3, 2.8e-7, 3.5e2, 4.8e-1, 2.947891711786755e-11),
                  (-1.2, 2.8e-6, 5.7e0, 4.6e-2, 0.66008237200814002),
                  (-0.25, 4.5e-7, 5.4e-2, 3.3e-4, 49.682658097740642),
                  (0.58, 2.8e-1, 3.1e1, 9.7e0, 0.025994957796295248),
                  (1.67, 4.1e-7, 4.9e-2, 5.6e-5, 0.00066552095239648704),
                  (0.72, 5.1e-7, 8.3e1, 1.2e2, 0.0),
                  (1.42, 3.6e-4, 9.5e-2, 2.37e-6, 0.0),
                  (4.1, 7.5e-4, 4.36e-1, -9.8e3, 0.0)]
    for idx, (slope, low, high, x, y_expected) in enumerate(test_cases):
        values = np.array([x, slope, low, high], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('PowerLaw [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_power_law_falling(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[3],
                              __global cdouble ret_value[1]) {
        ret_value[0] = power_law_falling(values[0], values[1], values[2]);
        return;
    }
    """

    # - value**slope * (slope+1) / low**(slope+1)
    test_cases = [(-3, 6.6e-9, 5.1e-7, 656.76097428590867),
                  (-2, 9.4e-2, 6.4e-1, 0.2294921875),
                  (-2, 4.7e-2, 5.7e0, 0.0014465989535241603),
                  (-3, 2.8e-7, 3.5e-2, 3.6571428571428644e-09),
                  (-1.2, 2.8e-6, 5.7e-3, 7.6456035583349609),
                  (-1.67, 4.1e-7, 4.9e-2, 0.0054201653192023375),
                  (-1.42, 9.5e-2, 3.6e-4, 0.0),
                  (-4.1, 4.36e-1, 7.5e-4, 0.0)]
    for idx, (slope, low, x, y_expected) in enumerate(test_cases):
        values = np.array([x, slope, low], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('PowerLawFalling [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_power_law(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[4],
                              __global cdouble ret_value[1]) {
        ret_value[0] = log_power_law(values[0], values[1], values[2], values[3]);
        return;
    }
    """

    # If slope == -1: - log(value) - log(log(high) - log(low))
    # If slope < -1: slope * log(value) + log(-slope-1) - log(low**(slope+1) - high**(slope+1))
    # else: slope * log(value) + log(slope+1) - log(high**(slope+1) - low**(slope+1))
    test_cases = [(-3, 6.6e-9, 5.1e0, 4.4e-3, -20.700792981094352),
                  (-2, 9.4e-2, 6.4e1, 9.2e0, -6.8013976351515346),
                  (-1, 5.8e-8, 3.8e2, 5.4e1, -7.1070664254446418),
                  (0, 1.5e-8, 8.4e1, 5.2e-4, -4.4308167986647424),
                  (1, 1.2e-5, 6.1e-3, 7.9e-4, 3.7486064535974082),
                  (2, 4.7e-2, 5.7e1, 2.4e1, -4.6744338535790284),
                  (3, 2.8e-7, 3.5e2, 4.8e-1, -24.247345782054548),
                  (-1.2, 2.8e-6, 5.7e0, 4.6e-2, -0.41539064567635631),
                  (-0.25, 4.5e-7, 5.4e-2, 3.3e-4, 3.9056559405767781),
                  (0.58, 2.8e-1, 3.1e1, 9.7e0, -3.6498526906794071),
                  (1.67, 4.1e-7, 4.9e-2, 5.6e-5, -7.3149404369335187),
                  (0.72, 5.1e-7, 8.3e1, 1.2e2, -np.inf),
                  (1.42, 3.6e-4, 9.5e-2, 2.37e-6, -np.inf),
                  (4.1, 7.5e-4, 4.36e-1, -9.8e3, -np.inf)]
    for idx, (slope, low, high, x, y_expected) in enumerate(test_cases):
        values = np.array([x, slope, low, high], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        if np.isinf(y_expected):
            vprint('LogPowerLaw [{:>2}]: {:>13.6e} vs. {:>13.6e}'
                   ''.format(idx, y, y_expected))
            assert np.isinf(y)
            assert y < 0
            continue
        vprint('LogPowerLaw [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_power_law_falling(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[3],
                              __global cdouble ret_value[1]) {
        ret_value[0] = log_power_law_falling(values[0], values[1], values[2]);
        return;
    }
    """

    # slope * log(value) + log(-slope-1) - (slope+1) * log(low)
    test_cases = [(-3, 6.6e-9, 5.1e-7, 6.4873201384160026),
                  (-2, 9.4e-2, 6.4e-1, -1.4718862914552941),
                  (-2, 4.7e-2, 5.7e0, -6.538540026953088),
                  (-3, 2.8e-7, 3.5e-2, -19.426583634516206),
                  (-1.2, 2.8e-6, 5.7e-3, 2.034130784379236),
                  (-1.67, 4.1e-7, 4.9e-2, -5.2176289622979013),
                  (-1.42, 9.5e-2, 3.6e-4, -np.inf),
                  (-4.1, 4.36e-1, 7.5e-4, -np.inf)]
    for idx, (slope, low, x, y_expected) in enumerate(test_cases):
        values = np.array([x, slope, low], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        if np.isinf(y_expected):
            vprint('LogPowerLawFalling [{:>2}]: {:>13.6e} vs. {:>13.6e}'
                   ''.format(idx, y, y_expected))
            assert np.isinf(y)
            assert y < 0
            continue
        vprint('LogPowerLawFalling [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


# %% Distribution integrals

def test_gaussian_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = gaussian(values[i], meanval, sigmaval);
        }
        return;
    }
    """

    test_cases = [(6400, 0, 1, -100, 100),
                  (6400, 42.1, 0.4, 30, 45.7),
                  (6400, -75.4, 13.6, -300, 100)]
    for idx, (N, mean, sigma, minval, maxval) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define meanval {}
        #define sigmaval {}
        """.format(N, mean, sigma)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(y, x)
        vprint('Gaussian Integral [{:>2}]: {:>13.6e}'
               ''.format(idx, integrated - 1))
        assert close(integrated, 1, tolerance)
    return


def test_log_gaussian_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = log_gaussian(values[i], meanval, sigmaval);
        }
        return;
    }
    """

    test_cases = [(6400, 0, 1, -100, 100),
                  (6400, 42.1, 0.4, 30, 45.7),
                  (6400, -75.4, 13.6, -300, 100)]
    for idx, (N, mean, sigma, minval, maxval) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define meanval {}
        #define sigmaval {}
        """.format(N, mean, sigma)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(np.exp(y), x)
        vprint('LogGaussian Integral [{:>2}]: {:>13.6e}'
               ''.format(idx, integrated - 1))
        assert close(integrated, 1, tolerance)
    return


def test_trunc_gaussian_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = trunc_gaussian(values[i], meanval, sigmaval, low, high);
        }
        return;
    }
    """

    test_cases = [(128000, 0, 1, -0.8, 0.92, -1, 1),
                  (128000, 42.1, 0.4, 32, 44, 30, 45.7),
                  (128000, -75.4, 13.6, -82.1, -76.9, -100, -50)]
    for idx, (N, mean, sigma, low, high, minval, maxval) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define meanval {}
        #define sigmaval {}
        #define low {}
        #define high {}
        """.format(N, mean, sigma, low, high)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(y, x)
        vprint('TruncGaussian Integral [{:>2}]: {:>13.6e}'
               ''.format(idx, integrated - 1))
        # TruncNormal seems to be quite hard to integrate using trapz,
        # so we use a much higher number of points and more lenient
        # tolerance. Note that this limitation should be largely
        # independent of limitations based on the float precision.
        assert close(integrated, 1, 1e-4)
    return


def test_log_trunc_gaussian_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = log_trunc_gaussian(values[i], meanval, sigmaval, low, high);
        }
        return;
    }
    """

    test_cases = [(128000, 0, 1, -0.8, 0.92, -1, 1),
                  (128000, 42.1, 0.4, 32, 44, 30, 45.7),
                  (128000, -75.4, 13.6, -82.1, -76.9, -100, -50)]
    for idx, (N, mean, sigma, low, high, minval, maxval) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define meanval {}
        #define sigmaval {}
        #define low {}
        #define high {}
        """.format(N, mean, sigma, low, high)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(np.exp(y), x)
        vprint('LogTruncGaussian Integral [{:>2}]: {:>13.6e}'
               ''.format(idx, integrated - 1))
        # TruncNormal seems to be quite hard to integrate using trapz,
        # so we use a much higher number of points and more lenient
        # tolerance. Note that this limitation should be largely
        # independent of limitations based on the float precision.
        assert close(integrated, 1, 1e-4)
    return


def test_power_law_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = power_law(values[i], slope, low, high);
        }
        return;
    }
    """

    test_cases = [(128000, -4, 1, 10, 0.5, 12),
                  (128000, -3, 1, 10, 0.5, 12),
                  (128000, -2, 1, 10, 0.5, 12),
                  (128000, -1, 1, 10, 0.5, 12),
                  (128000, 0, 1, 10, 0.5, 12),
                  (128000, 1, 1, 10, 0.5, 12),
                  (128000, 2, 1, 10, 0.5, 12),
                  (128000, 3, 1, 10, 0.5, 12),
                  (128000, 4, 1, 10, 0.5, 12),
                  (128000, -0.13, 1e-3, 1.2, 0.9e-3, 2),
                  (128000, 0.78, 14.6, 5.31e2, 12, 1e3),
                  (128000, -1.23, 5e-5, 3.2e-3, 5e-5, 3.2e-3),
                  (128000, 1.64, 1.64, 2.64, 1.5, 2.7)]
    for idx, (N, slope, low, high, minval, maxval) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define slope {}
        #define low {}
        #define high {}
        """.format(N, slope, low, high)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(y, x)
        vprint('PowerLaw Integral [{:>2}]: {:>13.6e}'
               ''.format(idx, integrated - 1))
        # Similar to trunc_normal, integrating past the limits drops the
        # accuracy of trapz massively. For confirmation, we include one
        # test case where integration and definition bounds coincide.
        assert close(integrated, 1, 1e-4)
    return


def test_power_law_falling_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = power_law_falling(values[i], slope, low);
        }
        return;
    }
    """

    # expected: 1 - (maxval / low)**(slope+1)
    test_cases = [(128000, -4, 1, 0.5, 8, 0.998046875),
                  (128000, -3, 1, 0.5, 12, 0.9930555555555556),
                  (128000, -2, 1, 0.5, 9, 0.8888888888888888),
                  (128000, -1.13, 1e-1, 0.9e-3, 27, 0.5170271580077825),
                  (128000, -1.78, 14.6, 12, 1e3, 0.9630005534740876),
                  (128000, -1.23, 5e-5, 5e-5, 3.2e-3, 0.6157812046779969)]
    for idx, (N, slope, low, minval, maxval, expected) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define slope {}
        #define low {}
        """.format(N, slope, low)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(y, x)
        vprint('PowerLawFalling Integral [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, integrated, expected, integrated - expected))
        # Similar to trunc_normal, integrating past the limits drops the
        # accuracy of trapz massively. For confirmation, we include one
        # test case where integration and definition bounds coincide.
        assert close(integrated, expected, 1e-4)
    return


def test_log_power_law_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = log_power_law(values[i], slope, low, high);
        }
        return;
    }
    """

    test_cases = [(128000, -4, 1, 10, 0.5, 12),
                  (128000, -3, 1, 10, 0.5, 12),
                  (128000, -2, 1, 10, 0.5, 12),
                  (128000, -1, 1, 10, 0.5, 12),
                  (128000, 0, 1, 10, 0.5, 12),
                  (128000, 1, 1, 10, 0.5, 12),
                  (128000, 2, 1, 10, 0.5, 12),
                  (128000, 3, 1, 10, 0.5, 12),
                  (128000, 4, 1, 10, 0.5, 12),
                  (128000, -0.13, 1e-3, 1.2, 0.9e-3, 2),
                  (128000, 0.78, 14.6, 5.31e2, 12, 1e3),
                  (128000, -1.23, 5e-5, 3.2e-3, 5e-5, 3.2e-3),
                  (128000, 1.64, 1.64, 2.64, 1.5, 2.7)]
    for idx, (N, slope, low, high, minval, maxval) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define slope {}
        #define low {}
        #define high {}
        """.format(N, slope, low, high)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(np.exp(y), x)
        vprint('LogPowerLaw Integral [{:>2}]: {:>13.6e}'
               ''.format(idx, integrated - 1))
        # Similar to trunc_normal, integrating past the limits drops the
        # accuracy of trapz massively. For confirmation, we include one
        # test case where integration and definition bounds coincide.
        assert close(integrated, 1, 1e-4)
    return


def test_log_power_law_falling_integral(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_value[i] = log_power_law_falling(values[i], slope, low);
        }
        return;
    }
    """

    # expected: 1 - (maxval / low)**(slope+1)
    test_cases = [(128000, -4, 1, 0.5, 8, 0.998046875),
                  (128000, -3, 1, 0.5, 12, 0.9930555555555556),
                  (128000, -2, 1, 0.5, 9, 0.8888888888888888),
                  (128000, -1.13, 1e-1, 0.9e-3, 27, 0.5170271580077825),
                  (128000, -1.78, 14.6, 12, 1e3, 0.9630005534740876),
                  (128000, -1.23, 5e-5, 5e-5, 3.2e-3, 0.6157812046779969)]
    for idx, (N, slope, low, minval, maxval, expected) in enumerate(test_cases):
        x = np.linspace(minval, maxval, N)
        defines = """
        #define N {}
        #define slope {}
        #define low {}
        """.format(N, slope, low)
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[N],
                                   kernel_name='test_kernel')[0]
        integrated = scipy.integrate.trapz(np.exp(y), x)
        vprint('LogPowerLawFalling Integral [{:>2}]: {:>13.6e} vs. {:>13.6e} ({})'
               ''.format(idx, integrated, expected, integrated - expected))
        # Similar to trunc_normal, integrating past the limits drops the
        # accuracy of trapz massively. For confirmation, we include one
        # test case where integration and definition bounds coincide.
        assert close(integrated, expected, 1e-4)
    return


# %% Virual confirmation for distribution shapes

def visual_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = gaussian(values[i], meanval, sigmaval);
        }
        return;
    }
    """

    mean = np.random.uniform(-256, 256)
    sigma = 10**np.random.uniform(-2, 2)
    minval = mean - (3*sigma * np.random.uniform(0.75, 1.5))
    maxval = mean + (3*sigma * np.random.uniform(0.75, 1.5))
    x = np.linspace(minval, maxval, 720)
    y_expected = scipy.stats.norm.pdf(x, loc=mean, scale=sigma)
    defines = """
    #define N {}
    #define meanval {}
    #define sigmaval {}
    """.format(len(x), mean, sigma)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, '--', label='Scipy')
    plt.legend()
    plt.title('Gaussian\nMean={:.2}, Sigma={:.2}'.format(mean, sigma))
    plt.tight_layout()
    return


def visual_log_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = log_gaussian(values[i], meanval, sigmaval);
        }
        return;
    }
    """

    mean = np.random.uniform(-256, 256)
    sigma = 10**np.random.uniform(-2, 2)
    minval = mean - (3*sigma * np.random.uniform(0.75, 1.5))
    maxval = mean + (3*sigma * np.random.uniform(0.75, 1.5))
    x = np.linspace(minval, maxval, 720)
    y_expected = scipy.stats.norm.logpdf(x, loc=mean, scale=sigma)
    defines = """
    #define N {}
    #define meanval {}
    #define sigmaval {}
    """.format(len(x), mean, sigma)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, '--', label='Scipy')
    plt.legend()
    plt.title('LogGaussian\nMean={:.2}, Sigma={:.2}'.format(mean, sigma))
    plt.tight_layout()
    return


def visual_trunc_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = trunc_gaussian(values[i], meanval, sigmaval, low, high);
        }
        return;
    }
    """

    mean = np.random.uniform(-256, 256)
    sigma = 10**np.random.uniform(-2, 2)
    minval = mean - (3*sigma * np.random.uniform(0.75, 1.5))
    maxval = mean + (3*sigma * np.random.uniform(0.75, 1.5))
    low = np.random.uniform(minval, mean)
    high = np.random.uniform(mean, maxval)
    x = np.linspace(minval, maxval, 720)
    a, b = (low - mean) / sigma, (high - mean) / sigma
    y_expected = scipy.stats.truncnorm.pdf(x, a, b, mean, sigma)
    defines = """
    #define N {}
    #define meanval {}
    #define sigmaval {}
    #define low {}
    #define high {}
    """.format(len(x), mean, sigma, low, high)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, '--', label='Scipy')
    plt.legend()
    plt.title('TruncGaussian\nMean={:.2}, Sigma={:.2}, Low={:.2}, High={:.2}'
              ''.format(mean, sigma, low, high))
    plt.tight_layout()
    return


def visual_log_trunc_gaussian(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = log_trunc_gaussian(values[i], meanval, sigmaval, low, high);
        }
        return;
    }
    """

    mean = np.random.uniform(-256, 256)
    sigma = 10**np.random.uniform(-2, 2)
    minval = mean - (3*sigma * np.random.uniform(0.75, 1.5))
    maxval = mean + (3*sigma * np.random.uniform(0.75, 1.5))
    low = np.random.uniform(minval, mean)
    high = np.random.uniform(mean, maxval)
    x = np.linspace(minval, maxval, 720)
    a, b = (low - mean) / sigma, (high - mean) / sigma
    y_expected = scipy.stats.truncnorm.logpdf(x, a, b, mean, sigma)
    defines = """
    #define N {}
    #define meanval {}
    #define sigmaval {}
    #define low {}
    #define high {}
    """.format(len(x), mean, sigma, low, high)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    # At this point we replace the negative infinities with a small
    # but constant value, while not so small to completely distort
    # the plot.
    y[np.isinf(y)] = 1.5*np.min(y[~np.isinf(y)])
    y_expected[np.isinf(y_expected)] = 1.5*np.min(y_expected[~np.isinf(y_expected)])

    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, '--', label='Scipy')
    plt.legend()
    plt.title('LogTruncGaussian\nMean={:.2}, Sigma={:.2}, Low={:.2}, High={:.2}'
              ''.format(mean, sigma, low, high))
    plt.tight_layout()
    return


def visual_power_law(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = power_law(values[i], slope, low, high);
        }
        return;
    }
    """

    slope = np.random.uniform(-2, 3)
    minval = 10**np.random.uniform(-3, 3)
    maxval = minval * np.random.uniform(2160, 4096)
    low = minval * np.random.uniform(1, 1.2)
    high = maxval * np.random.uniform(0.8, 1)
    x = np.linspace(minval, maxval, 7200)
    y_expected = x**slope * (slope+1) / (high**(slope+1) - low**(slope+1))
    y_expected[np.logical_or(x < low, x > high)] = 0
    defines = """
    #define N {}
    #define slope {}
    #define low {}
    #define high {}
    """.format(len(x), slope, low, high)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.loglog(x, y, label='GAPS')
    plt.loglog(x, y_expected, '--', label='Python')
    plt.grid()
    plt.legend()
    plt.title('PowerLaw\nSlope={:.2}, Low={:.2}, High={:.2}'
              ''.format(slope, low, high))
    plt.tight_layout()
    return


def visual_power_law_falling(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = power_law_falling(values[i], slope, low);
        }
        return;
    }
    """

    slope = np.random.uniform(-3, -1)
    minval = 10**np.random.uniform(-3, 3)
    maxval = minval * np.random.uniform(2160, 4096)
    low = minval * np.random.uniform(1, 1.2)
    x = np.linspace(minval, maxval, 7200)
    y_expected = x**slope * (-slope-1) / low**(slope+1)
    y_expected[x < low] = 0
    defines = """
    #define N {}
    #define slope {}
    #define low {}
    """.format(len(x), slope, low)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.loglog(x, y, label='GAPS')
    plt.loglog(x, y_expected, '--', label='Python')
    plt.grid()
    plt.legend()
    plt.title('PowerLawFalling\nSlope={:.2}, Low={:.2}'.format(slope, low))
    plt.tight_layout()
    return


def visual_log_power_law(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = log_power_law(values[i], slope, low, high);
        }
        return;
    }
    """

    slope = np.random.uniform(-2, 3)
    minval = 10**np.random.uniform(-3, 3)
    maxval = minval * np.random.uniform(2160, 4096)
    low = minval * np.random.uniform(1, 1.2)
    high = maxval * np.random.uniform(0.8, 1)
    x = np.linspace(minval, maxval, 7200)
    y_expected = np.log(x**slope * (slope+1) / (high**(slope+1) - low**(slope+1)))
    y_expected[np.logical_or(x < low, x > high)] = -np.inf
    defines = """
    #define N {}
    #define slope {}
    #define low {}
    #define high {}
    """.format(len(x), slope, low, high)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.semilogx(x, y, label='GAPS')
    plt.semilogx(x, y_expected, '--', label='Python')
    plt.grid()
    plt.legend()
    plt.title('LogPowerLaw (base e)\nSlope={:.2}, Low={:.2}, High={:.2}'
              ''.format(slope, low, high))
    plt.tight_layout()
    return


def visual_log_power_law_falling(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_values[N]) {
        for(size_t i = 0; i < N; i++) {
            ret_values[i] = log_power_law_falling(values[i], slope, low);
        }
        return;
    }
    """

    slope = np.random.uniform(-3, -1)
    minval = 10**np.random.uniform(-3, 3)
    maxval = minval * np.random.uniform(2160, 4096)
    low = minval * np.random.uniform(1, 1.2)
    x = np.linspace(minval, maxval, 7200)
    y_expected = slope*np.log(x) + np.log(-slope-1) - (slope+1)*np.log(low)
    y_expected[x < low] = -np.inf
    defines = """
    #define N {}
    #define slope {}
    #define low {}
    """.format(len(x), slope, low)
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[len(x)],
                               kernel_name='test_kernel')[0]
    plt.figure()
    plt.semilogx(x, y, label='GAPS')
    plt.semilogx(x, y_expected, '--', label='Python')
    plt.grid()
    plt.legend()
    plt.title('LogPowerLawFalling (base e)\nSlope={:.2}, Low={:.2}'
              ''.format(slope, low))
    plt.tight_layout()
    return


if __name__ == '__main__':
    VISUAL = True
    VERBOSE = True
    import matplotlib.pyplot as plt

    for arguments in device_list:
        if (arguments[0], arguments[2]) != (0, 1):
            continue
        print(arguments[1].name, arguments[3].name, sep=' - ')
        test_math_function_product(arguments)
        test_math_function_sum(arguments)
        test_math_function_logsumexp(arguments)
        test_math_function_logaddexp(arguments)
        test_math_function_mean(arguments)
        test_math_function_stddev(arguments)
        test_math_function_iter_min(arguments)
        test_math_function_iter_max(arguments)

        test_gaussian(arguments)
        test_log_gaussian(arguments)
        test_trunc_gaussian(arguments)
        test_log_trunc_gaussian(arguments)
        test_power_law(arguments)
        test_power_law_falling(arguments)
        test_log_power_law(arguments)
        test_log_power_law_falling(arguments)

        test_gaussian_integral(arguments)
        test_log_gaussian_integral(arguments)
        test_trunc_gaussian_integral(arguments)
        test_log_trunc_gaussian_integral(arguments)
        test_power_law_integral(arguments)
        test_power_law_falling_integral(arguments)
        test_log_power_law_integral(arguments)
        test_log_power_law_falling_integral(arguments)

        visual_gaussian(arguments)
        visual_log_gaussian(arguments)
        visual_trunc_gaussian(arguments)
        visual_log_trunc_gaussian(arguments)
        visual_power_law(arguments)
        visual_power_law_falling(arguments)
        visual_log_power_law(arguments)
        visual_log_power_law_falling(arguments)
