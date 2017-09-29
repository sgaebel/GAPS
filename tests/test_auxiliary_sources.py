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

import gaps
import numpy as np
import pyopencl as ocl
import scipy.integrate
import scipy.misc
import scipy.stats

READ_ONLY = ocl.mem_flags.READ_ONLY
WRITE_ONLY = ocl.mem_flags.WRITE_ONLY
COPY_HOST_PTR = ocl.mem_flags.COPY_HOST_PTR

VISUAL = None


def device_iterator():
    for platform_idx, platform in enumerate(ocl.get_platforms()):
        for device_idx in range(len(platform.get_devices())):
            yield platform_idx, device_idx
    return


def type_and_tolerance(platform_idx, device_idx):
    device = ocl.get_platforms()[platform_idx].get_devices()[device_idx]
    if 'fp64' in device.get_info(ocl.device_info.EXTENSIONS):
        return np.float64, 1e-8
    else:
        return np.float32, 1e-4


def direct_evaluation(plat_idx, dev_idx, source_code, read_only, write_only):
    context, queue = gaps.create_context_and_queue(platform_idx=plat_idx,
                                                   device_idx=dev_idx)
    read_only = [buffer.astype(gaps._cdouble(queue)) for buffer in read_only]
    write_only = [buffer.astype(gaps._cdouble(queue)) for buffer in write_only]
    read_buffers = [ocl.Buffer(context, READ_ONLY | COPY_HOST_PTR, hostbuf=x)
                    for x in read_only]
    write_buffers = [ocl.Buffer(context, WRITE_ONLY, x.nbytes)
                     for x in write_only]
    buffers = read_buffers + write_buffers
    program = ocl.Program(context, gaps._basic_code(queue)+source_code).build()
    event = program.test_function(queue, (1,), (1,), *buffers)
    ocl.enqueue_barrier(queue, wait_for=[event])

    for idx, wo_buffer in enumerate(write_buffers):
        ocl.enqueue_copy(queue, write_only[idx], wo_buffer)
    return write_only


def test_math_function_results():
    for platform_idx, device_idx in device_iterator():
        cdouble, tolerance = type_and_tolerance(platform_idx, device_idx)
        values = np.random.uniform(0, 42, size=12)
        return_values = np.empty(8)
        defines = '#define N_VALUES {}'.format(len(values))
        kernel_source = """
        __kernel void test_function(__global const cdouble values[N_VALUES],
                                    __global cdouble return_values[8]) {
            return_values[0] = product(values, N_VALUES);
            return_values[1] = sum(values, N_VALUES);
            return_values[2] = logsumexp(values, N_VALUES);
            return_values[3] = logaddexp(values[0], values[N_VALUES-1]);
            return_values[4] = mean(values, N_VALUES);
            return_values[5] = stddev(values, N_VALUES);
            return_values[6] = iter_min(values, N_VALUES);
            return_values[7] = iter_max(values, N_VALUES);
        }
        """
        source_code = defines + kernel_source
        return_values = direct_evaluation(platform_idx, device_idx,
                                          source_code,
                                          read_only=[values],
                                          write_only=[return_values])[0]
        assert abs((return_values[0] - np.product(values))/np.product(values)) < tolerance
        assert abs(return_values[1] - np.sum(values)) < tolerance
        assert abs(return_values[2] - scipy.misc.logsumexp(values)) < tolerance
        assert abs(return_values[3] - np.logaddexp(values[0], values[-1])) < tolerance
        assert abs(return_values[4] - np.mean(values)) < tolerance
        assert abs(return_values[5] - np.std(values)) < tolerance
        assert abs(return_values[6] - np.min(values)) < tolerance
        assert abs(return_values[7] - np.max(values)) < tolerance


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

#    test_math_function_results()
#    test_gaussian_normed_pdf_values()
#    test_gaussian_normed_pdf_integral()
#    test_log_gaussian_normed_pdf_values()
#    test_gaussian_pdf_values()
#    test_log_gaussian_pdf_values()
#    test_trunc_gaussian_pdf_values()
#    test_trunc_gaussian_pdf_integral()
#    test_log_trunc_gaussian_pdf_values()
    test_trunc_gaussian_pdf_values()
    test_trunc_gaussian_pdf_integral()
    test_log_trunc_gaussian_pdf_values()
#    test_power_law_pdf_values()
#    test_power_law_pdf_integral()
#    test_power_law_falling_pdf_values()
#    test_log_power_law_pdf_values()
#    test_log_power_law_falling_pdf_values()


