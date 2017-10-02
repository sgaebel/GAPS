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


def type_and_tolerance(platform_idx, device_idx):
    device = ocl.get_platforms()[platform_idx].get_devices()[device_idx]
    if 'fp64' in device.get_info(ocl.device_info.EXTENSIONS):
        return np.float64, 1e-8
    else:
        return np.float32, 1e-4


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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.product(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        #print(f'Product [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
    return


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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.sum(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        #print(f'Sum [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = scipy.misc.logsumexp(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        #print(f'LogSumExp [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 2).astype(cdouble)
        y_expected = np.logaddexp(x[0], x[1])
        y = gaps.direct_evaluation(kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        #print(f'LogAddExp [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.mean(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        #print(f'Mean [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.std(x, dtype=cdouble)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        #print(f'StdDev [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.min(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        print(f'Min [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
    for idx in range(10):
        x = np.random.uniform(1e-4, 1e4, 10).astype(cdouble)
        y_expected = np.max(x)
        y = gaps.direct_evaluation(f'#define N {len(x)}' + kernel_source,
                                   platform_idx=platform_idx,
                                   device_idx=device_idx,
                                   read_only_arrays=[x],
                                   write_only_shapes=[1],
                                   kernel_name='test_kernel')[0][0]
        print(f'Max [{idx:>2}]: {y:>12.6f} vs. {y_expected:>12.6f} ({y-y_expected})')
        assert abs(y - y_expected) < tolerance
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
        test_math_function_product(args)
        test_math_function_sum(args)
        test_math_function_logsumexp(args)
        test_math_function_logaddexp(args)
        test_math_function_mean(args)
        test_math_function_stddev(args)
        test_math_function_iter_min(args)
        test_math_function_iter_max(args)

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
