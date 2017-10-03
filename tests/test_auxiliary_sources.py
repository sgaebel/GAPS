#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for Python 3

@author: Sebastian Gaebel
@email: sgaebel@star.sr.bham.ac.uk
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
VERBOSE = True
N_MATH_TESTS = 3


"""
@pytest.fixture(params=[model1, model2, model3])
def model(request):
    return request.param

def test_awesome(model):
    assert model == "awesome"








def pytest_generate_tests(metafunc):
    if "model" in metafunc.funcargnames:
        models = [model1,model2,model3]
        for model in models:
            metafunc.addcall(funcargs=dict(model=model))

def test_awesome(model):
    assert model == "awesome"
"""


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

    # TODO: hardcode important tests
    # e.g.: include a zero, have some negative numbers
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.sum(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Sum [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # e.g.: include a zero, have some negative numbers
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.product(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Product [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = scipy.misc.logsumexp(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('LogSumExp [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 2).astype(cdouble)
        y_expected = np.logaddexp(x[0], x[1])
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('LogAddExp [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.mean(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Mean [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.std(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('StdDev [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.min(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Min [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # TODO: hardcode important tests
    # also check against fmax
    for idx in range(N_MATH_TESTS):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Max [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # Expected values: scipy.stats.norm.pdf(x, mean, sigma) * (sigma * np.sqrt(2*np.pi))
    test_cases = [(42, 10.1, 31.2, 0.56455997531486624),
                  (-12.4, 35, 5.6, 0.87612640723526436),
                  (0, 1, 0, 1.0),
                  (0, 1, 1, 0.60653065971263342),
                  (-2, 0.2, 1, 1.3863432936411706e-49),
                  (-2, 0.2, -12, 0.0)]
    for idx, (mean, sigma, x, y_expected) in enumerate(test_cases):
        values = np.array([mean, sigma, x], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Gaussian [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_gaussian_normed(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[3],
                              __global cdouble ret_value[1]) {
        ret_value[0] = gaussian_normed(values[2], values[0], values[1]);
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
        vprint('GaussianNormed [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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

    # Expected values: -0.5 * (value - mean)**2 / sigma**2
    test_cases = [(42, 10.1, 31.2, -0.5717086560141164),
                  (-12.4, 35, 5.6, -0.13224489795918368),
                  (0, 1, 0, 0.0),
                  (0, 1, 1, -0.5),
                  (-2, 0.2, 1, -112.49999999999997),
                  (-2, 0.2, -12, -1249.9999999999998)]
    for idx, (mean, sigma, x, y_expected) in enumerate(test_cases):
        values = np.array([mean, sigma, x], dtype=cdouble)
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[values],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('LogGaussian [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_gaussian_normed(args):
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[3],
                              __global cdouble ret_value[1]) {
        ret_value[0] = log_gaussian_normed(values[2], values[0], values[1]);
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
        vprint('LogGaussianNormed [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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
        vprint('TruncGaussian [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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
            vprint('LogTruncGaussian [{:>2}]: {:>12.6e} vs. {:>12.6e}'
                   ''.format(idx, y, y_expected))
            assert np.isinf(y)
            assert y < 0
            continue
        vprint('LogTruncGaussian [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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
        vprint('PowerLaw [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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
        vprint('PowerLawFalling [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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
            vprint('LogPowerLaw [{:>2}]: {:>12.6e} vs. {:>12.6e}'
                   ''.format(idx, y, y_expected))
            assert np.isinf(y)
            assert y < 0
            continue
        vprint('LogPowerLaw [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
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
            vprint('LogPowerLawFalling [{:>2}]: {:>12.6e} vs. {:>12.6e}'
                   ''.format(idx, y, y_expected))
            assert np.isinf(y)
            assert y < 0
            continue
        vprint('LogPowerLawFalling [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


# %% Distribution integrals

def test_gaussian_normed_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_gaussian_normed_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_trunc_gaussian_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_trunc_gaussian_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_power_law_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_power_law_falling_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_power_law_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


def test_log_power_law_falling_integral(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    test_cases = [(parameter1, parameter2, x_value, y_value)]
    for some_args in test_cases:
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        defines = """
        #define N {len(x)}
        """.format()
        y = gaps.direct_evaluation(defines + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        vprint('Whatever [{:>2}]: {:>12.6e} vs. {:>12.6e} ({})'
               ''.format(idx, y, y_expected, y-y_expected))
        assert close(y, y_expected, tolerance)
    return


# %% Virual confirmation for distribution shapes

def visual_gaussian(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_gaussian_normed(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_log_gaussian(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_log_gaussian_normed(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_trunc_gaussian(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_log_trunc_gaussian(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_power_law(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_power_law_falling(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_log_power_law(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return


def visual_log_power_law_falling(args):
    raise NotImplementedError
    platform_idx, platform, device_idx, device = args
    cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
    kernel_source = """
    __kernel void test_kernel(__global const cdouble values[N],
                              __global cdouble ret_value[1]) {
        ret_value[0] = some_function(argument list);
        return;
    }
    """

    # TODO: hardcode important tests
    # also check against fmax
    some_args = (parameter1, parameter2, x_value, y_value)
    x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
    y_expected = np.max(x)
    defines = """
    #define N {len(x)}
    """.format()
    y = gaps.direct_evaluation(defines + kernel_source,
                               platform_idx=platform_idx,
                               device_idx=device_idx,
                               read_only_arrays=[x],
                               write_only_shapes=[1],
                               kernel_name='test_kernel')[0][0]
    plt.figure()
    plt.plot(x, y, label='GAPS')
    plt.plot(x, y_expected, ':', label='Scipy')
    plt.legend()
    plt.title('Something about what we expected and the float type')
    plt.tight_layout()
    return






# %% Old Tests


def test_gaussian_normed_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}""".format(len(x), mean, sigma)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = gaussian_normed(x[i], meanvalue, sigmavalue);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = scipy.stats.norm.pdf(x, mean, sigma)
        if VISUAL:
            plt.figure()
            plt.plot(x, y, label='GAPS')
            plt.plot(x, y_expected, ':', label='Scipy')
            plt.legend()
            plt.title('test_gaussian_normed_pdf_values\n({}.{}), {}'
                      ''.format(platform_idx, device_idx, cdouble))
            plt.tight_layout()
            plt.show()
        assert np.all(np.abs(y_expected - y) < tolerance)




def test_gaussian_normed_pdf_integral():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}""".format(len(x), mean, sigma)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = gaussian_normed(x[i], meanvalue, sigmavalue);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = scipy.stats.norm.pdf(x, mean, sigma)
        assert abs(scipy.integrate.trapz(y, x) - 1) < 100*tolerance
        assert abs(scipy.integrate.trapz(y_expected, x) - 1) < 100*tolerance


def test_log_gaussian_normed_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}""".format(len(x), mean, sigma)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = log_gaussian_normed(x[i], meanvalue, sigmavalue);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = scipy.stats.norm.logpdf(x, mean, sigma)
        if VISUAL:
            plt.figure()
            plt.plot(x, y, label='GAPS')
            plt.plot(x, y_expected, ':', label='Scipy')
            plt.legend()
            plt.title('test_log_gaussian_normed_pdf_values\n({}.{}), {}'
                      ''.format(platform_idx, device_idx, cdouble))
            plt.tight_layout()
            plt.show()
        assert np.all(np.abs(y - y_expected) < tolerance)


def test_gaussian_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}""".format(len(x), mean, sigma)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = gaussian(x[i], meanvalue, sigmavalue);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = scipy.stats.norm.pdf(x, mean, sigma)
        if VISUAL:
            plt.figure()
            plt.plot(x, y, label='GAPS')
            plt.plot(x, y_expected, ':', label='Scipy')
            plt.legend()
            plt.title('test_gaussian_pdf_values\n({}.{}), {}'
                      ''.format(platform_idx, device_idx, cdouble))
            plt.tight_layout()
            plt.show()
        assert np.std(y / y_expected) < tolerance


def test_log_gaussian_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}""".format(len(x), mean, sigma)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = log_gaussian(x[i], meanvalue, sigmavalue);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = scipy.stats.norm.logpdf(x, mean, sigma)
        if VISUAL:
            plt.figure()
            plt.plot(x, y, label='GAPS')
            plt.plot(x, y_expected, ':', label='Scipy')
            plt.legend()
            plt.title('test_log_gaussian_pdf_values\n({}.{}), {}'
                      ''.format(platform_idx, device_idx, cdouble))
            plt.tight_layout()
            plt.show()
        # Instead of checking the values directly, check that the offset
        # between normed and unnormed is very close to constant.
        assert np.std(np.abs(y - y_expected)) < tolerance


def test_trunc_gaussian_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}
        #define low {}
        #define high {}""".format(len(x), mean, sigma, low, high)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = trunc_gaussian(x[i], meanvalue, sigmavalue, low, high);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]

        a, b = (low - mean) / sigma, (high - mean) / sigma
        y_expected = scipy.stats.truncnorm.pdf(x, a, b, mean, sigma)
        if VISUAL:
            plt.figure()
            plt.plot(x, y, label='GAPS')
            plt.plot(x, y_expected, ':', label='Scipy')
            plt.legend()
            plt.title('test_gaussian_pdf_values\n({}.{}), {}'
                      ''.format(platform_idx, device_idx, cdouble))
            plt.tight_layout()
            plt.show()
            print(mean, sigma, low, high)
        assert np.std(y / y_expected) < tolerance


def test_trunc_gaussian_pdf_integral():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}
        #define low {}
        #define high {}""".format(len(x), mean, sigma, low, high)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = trunc_gaussian(x[i], meanvalue, sigmavalue, low, high);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]

        a, b = (low - mean) / sigma, (high - mean) / sigma
        y_expected = scipy.stats.truncnorm.pdf(x, a, b, loc=mean, scale=sigma)
        assert abs(scipy.integrate.trapz(y, x) - 1) < 100*tolerance
        assert abs(scipy.integrate.trapz(y_expected, x) - 1) < 100*tolerance


def test_log_trunc_gaussian_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        mean = np.random.uniform(-100, 100)
        sigma = 10**np.random.uniform(-2, 2)
        low = mean - sigma * np.random.uniform(5, 6)
        high = mean + sigma * np.random.uniform(5, 6)
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define meanvalue {}
        #define sigmavalue {}
        #define low {}
        #define high {}""".format(len(x), mean, sigma, low, high)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = log_trunc_gaussian(x[i], meanvalue, sigmavalue, low, high);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]

        a, b = (low - mean) / sigma, (high - mean) / sigma
        y_expected = scipy.stats.truncnorm.logpdf(x, a, b, loc=mean, scale=sigma)
        if VISUAL:
            plt.figure()
            plt.plot(x, y, label='GAPS')
            plt.plot(x, y_expected, ':', label='Scipy')
            plt.legend()
            plt.title('test_log_gaussian_pdf_values\n({}.{}), {}'
                      ''.format(platform_idx, device_idx, cdouble))
            plt.tight_layout()
            plt.show()
        # Instead of checking the values directly, check that the offset
        # between normed and unnormed is very close to constant.
        assert np.std(np.abs(y - y_expected)) < tolerance


def test_power_law_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        slope = np.random.uniform(-3, 3)
        low, high = sorted(10**np.random.uniform(-5, 5, 2))
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define slope {}
        #define low {}
        #define high {}""".format(len(x), slope, low, high)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = power_law(x[i], slope, low, high);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = x**slope * (slope+1) / (high**(slope+1) - low**(slope+1))
        # Instead of checking the values directly, check that the offset
        # between normed and unnormed is very close to constant.
        if VISUAL:
            plt.figure()
            plt.loglog(x, y, label='GAPS')
            plt.loglog(x, y_expected, ':', label='Scipy')
            plt.grid()
            plt.legend()
            plt.title('test_power_law_pdf_values\n({}.{}), {}, $\\alpha$ = {:.2f}'
                      ''.format(platform_idx, device_idx, cdouble, slope))
            plt.tight_layout()
            plt.show()
        assert np.all(np.abs(y - y_expected) < tolerance)


def test_power_law_pdf_integral():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        slope = np.random.uniform(-3, 3)
        low, high = sorted(10**np.random.uniform(-5, 5, 2))
        x = np.exp(np.linspace(np.log(low), np.log(high), num=6400))
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define slope {}
        #define low {}
        #define high {}""".format(len(x), slope, low, high)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = power_law(x[i], slope, low, high);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = x**slope * (slope+1) / (high**(slope+1) - low**(slope+1))
        assert abs(scipy.integrate.trapz(y, x) - 1) < 100*tolerance
        assert abs(scipy.integrate.trapz(y_expected, x) - 1) < 100*tolerance


def test_power_law_falling_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        slope = np.random.uniform(-5, -1.0001)
        low = 10**np.random.uniform(-5, 5)
        high = np.inf
        x = 10**np.linspace(np.log10(low), 12, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define slope {}
        #define low {}""".format(len(x), slope, low)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = power_law_falling(x[i], slope, low);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = x**slope * (slope+1) / (high**(slope+1) - low**(slope+1))
        # Instead of checking the values directly, check that the offset
        # between normed and unnormed is very close to constant.
        if VISUAL:
            plt.figure()
            plt.loglog(x, y, label='GAPS')
            plt.loglog(x, y_expected, ':', label='Scipy')
            plt.grid()
            plt.legend()
            plt.title('test_power_law_falling_pdf_values\n({}.{}), {}, $\\alpha$ = {:.2f}'
                      ''.format(platform_idx, device_idx, cdouble, slope))
            plt.tight_layout()
            plt.show()
        assert np.all((np.abs(y - y_expected)) < tolerance)


def test_log_power_law_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        slope = np.random.uniform(-3, 3)
        low, high = sorted(10**np.random.uniform(-5, 5, 2))
        x = np.linspace(low, high, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define slope {}
        #define low {}
        #define high {}""".format(len(x), slope, low, high)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = log_power_law(x[i], slope, low, high);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = np.log(x**slope * (slope+1) / (high**(slope+1) - low**(slope+1)))
        # Instead of checking the values directly, check that the offset
        # between normed and unnormed is very close to constant.
        if VISUAL:
            plt.figure()
            plt.semilogx(x, y, label='GAPS')
            plt.semilogx(x, y_expected, ':', label='Scipy')
            plt.grid()
            plt.legend()
            plt.title('test_power_law_pdf_values\n({}.{}), {}, $\\alpha$ = {:.2f}'
                      ''.format(platform_idx, device_idx, cdouble, slope))
            plt.tight_layout()
            plt.show()
        assert np.all(np.abs(y - y_expected) < tolerance)


def test_log_power_law_falling_pdf_values():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        slope = np.random.uniform(-5, -1.0001)
        low = 10**np.random.uniform(-5, 5)
        high = np.inf
        x = 10**np.linspace(np.log10(low), 12, num=6400)
        y = np.empty_like(x)
        defines = """
        #define N_VALUES {}
        #define slope {}
        #define low {}""".format(len(x), slope, low)
        kernel_source = """
        __kernel void test_function(__global const cdouble x[N_VALUES],
                                    __global cdouble y[N_VALUES]) {
            for(size_t i = 0; i < N_VALUES; i++) {
                y[i] = log_power_law_falling(x[i], slope, low);
            }
        }
        """
        source_code = defines + kernel_source
        y = direct_evaluation(platform_idx, device_idx, source_code,
                              read_only=[x], write_only=[y])[0]
        y_expected = np.log(x**slope * (slope+1) / (high**(slope+1) - low**(slope+1)))
        # Instead of checking the values directly, check that the offset
        # between normed and unnormed is very close to constant.
        if VISUAL:
            plt.figure()
            plt.semilogx(x, y, label='GAPS')
            plt.semilogx(x, y_expected, ':', label='Scipy')
            plt.grid()
            plt.legend()
            plt.title('test_power_law_falling_pdf_values\n({}.{}), {}, $\\alpha$ = {:.2f}'
                      ''.format(platform_idx, device_idx, cdouble, slope))
            plt.tight_layout()
            plt.show()
        assert np.all(np.abs(y - y_expected) < tolerance)


if __name__ == '__main__':
    VISUAL = True
    import matplotlib.pyplot as plt

    for args in device_list:
        if (args[0], args[2]) != (0, 1):
            continue
        print(args[1].name, args[3].name, sep=' - ')
#        test_math_function_product(args)
#        test_math_function_sum(args)
#        test_math_function_logsumexp(args)
#        test_math_function_logaddexp(args)
#        test_math_function_mean(args)
#        test_math_function_stddev(args)
#        test_math_function_iter_min(args)
#        test_math_function_iter_max(args)

#        test_gaussian(args)
#        test_gaussian_normed(args)
#        test_log_gaussian(args)
#        test_log_gaussian_normed(args)
#        test_trunc_gaussian(args)
#        test_log_trunc_gaussian(args)
#        test_power_law(args)
#        test_power_law_falling(args)
#        test_log_power_law(args)
#        test_log_power_law_falling(args)
#        test_gaussian_normed_integral(args)
#        test_log_gaussian_normed_integral(args)
#        test_trunc_gaussian_integral(args)
#        test_log_trunc_gaussian_integral(args)
#        test_power_law_integral(args)
#        test_power_law_falling_integral(args)
#        test_log_power_law_integral(args)
#        test_log_power_law_falling_integral(args)

        visual_gaussian(args)
        visual_gaussian_normed(args)
        visual_log_gaussian(args)
        visual_log_gaussian_normed(args)
        visual_trunc_gaussian(args)
        visual_log_trunc_gaussian(args)
        visual_power_law(args)
        visual_power_law_falling(args)
        visual_log_power_law(args)
        visual_log_power_law_falling(args)





    #    test_math_function_results()
    #    test_gaussian_normed_pdf_values()
    #    test_gaussian_normed_pdf_integral()
    #    test_log_gaussian_normed_pdf_values()
    #    test_gaussian_pdf_values()
    #    test_log_gaussian_pdf_values()
    #    test_trunc_gaussian_pdf_values()
    #    test_trunc_gaussian_pdf_integral()
    #    test_log_trunc_gaussian_pdf_values()
    #    test_trunc_gaussian_pdf_values()
    #    test_trunc_gaussian_pdf_integral()
    #    test_log_trunc_gaussian_pdf_values()
    #    test_power_law_pdf_values()
    #    test_power_law_pdf_integral()
    #    test_power_law_falling_pdf_values()
    #    test_log_power_law_pdf_values()
    #    test_log_power_law_falling_pdf_values()
        break
