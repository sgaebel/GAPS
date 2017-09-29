#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created for Python 3

@author: Sebastian Gaebel
@email: sgaebel@star.sr.bham.ac.uk
"""

import pyopencl as ocl
from .utilities import cdouble


# TODO: Other potentially useful stuff:
# * integration routine
# * interpolation in 1D, maybe more
# * KDE
# * median
# * percentile
# * sorting


# %% Math Macros
# TODO: Useful constants not included by default in OpenCL

math_constants = """
#define M_1_SQRTPI 0.56418958354775628
#define M_1_SQRT2PI 0.3989422804014327
#define M_LOG_1_SQRTPI -0.57236494292470008
#define M_LOG_1_SQRT2PI -0.91893853320467267
#define M_SQRT_2_PI 0.79788456080286541
#define M_LOG_SQRT_2_PI -0.22579135264472738
#define M_SQRT1_2 0.70710678118654757
"""


# %% Math templates

sum_template = """
cdouble sum(__global const cdouble * iterable, const size_t length) {
    cdouble accumulator = 0;
    for(size_t i = 0; i < length; i++) {
        accumulator += iterable[i];
    }
    return accumulator;
}
"""

product_template = """
cdouble product(__global const cdouble * iterable, const size_t length) {
    cdouble accumulator = 1;
    for(size_t i = 0; i < length; i++) {
        accumulator *= iterable[i];
    }
    return accumulator;
}
"""

logsumexp_template = """
cdouble logsumexp(__global const cdouble * log_values, size_t length) {
    cdouble max_value = -INFINITY;
    for(size_t i = 0; i < length; i++) {
        max_value = fmax(max_value, log_values[i]);
    }
    cdouble accumulator = 0;
    for(size_t i = 0; i < length; i++) {
        accumulator += exp(log_values[i] - max_value);
    }
    return log(accumulator) + max_value;
}
"""

logaddexp_template = """
cdouble logaddexp(const cdouble x, const cdouble y) {
    return fmax(x, y) + log1p(exp(-fabs(x - y)));
}
"""

mean_template = """
cdouble mean(__global const cdouble * iterable, const size_t length) {
    cdouble accumulator = 0;
    for(size_t i = 0; i < length; i++) {
        accumulator += iterable[i];
    }
    return accumulator / length;
}
"""

stddev_template = """
cdouble stddev(__global const cdouble * iterable, const size_t length) {
    const cdouble mean_value = mean(iterable, length);
    cdouble accumulator = 0;
    for(size_t i = 0; i < length; i++) {
        accumulator += pown(iterable[i] - mean_value, 2);
    }
    return sqrt(accumulator / length);
}
"""

min_template = """
cdouble iter_min(__global const cdouble * iterable, const size_t length) {
    cdouble current_min = iterable[0];
    for(size_t i = 1; i < length; i++) {
        current_min = fmin(current_min, iterable[i]);
    }
    return current_min;
}
"""

max_template = """
cdouble iter_max(__global const cdouble * iterable, const size_t length) {
    cdouble current_max = iterable[0];
    for(size_t i = 1; i < length; i++) {
        current_max = fmax(current_max, iterable[i]);
    }
    return current_max;
}
"""


# %% Distribution templates
# TODO: Add normed and/or log versions?
# TODO: Other distributions? chi-squared, lognormal
# TODO: Higher dimensions (esp. gaussian)?

gaussian_pdf_templates = """
cdouble gaussian(const cdouble value, const cdouble mean, const cdouble stddev) {
    return exp(-0.5 * pown((value - mean) / stddev, 2));
}

cdouble gaussian_normed(const cdouble value, const cdouble mean, const cdouble stddev) {
    return M_1_SQRT2PI * exp(-0.5 * pown((value - mean) / stddev, 2)) / stddev;
}

cdouble log_gaussian(const cdouble value, const cdouble mean, const cdouble stddev) {
    return -0.5 * pown((value - mean) / stddev, 2);
}

cdouble log_gaussian_normed(const cdouble value, const cdouble mean, const cdouble stddev) {
    return M_LOG_1_SQRT2PI - 0.5 * pown((value - mean) / stddev, 2) - log(stddev);
}
"""

trunc_gaussian_pdf_templates = """
cdouble trunc_gaussian(const cdouble value, const cdouble mean, const cdouble stddev,
                      const cdouble low, const cdouble high) {
    const cdouble inv_stddev = 1.0 / stddev;
    const cdouble erf_L = erf((low - mean) * M_SQRT1_2 * inv_stddev);
    const cdouble erf_H = erf((high - mean) * M_SQRT1_2 * inv_stddev);
    return M_SQRT_2_PI * inv_stddev * exp(-0.5 * pown((value - mean) * inv_stddev, 2)) / (erf_H - erf_L);
}

cdouble log_trunc_gaussian(const cdouble value, const cdouble mean, const cdouble stddev,
                          const cdouble low, const cdouble high) {
    const cdouble inv_stddev = 1.0 / stddev;
    const cdouble log_sqrt_2_pi = -0.22579135264472741;
    return log_sqrt_2_pi - log(stddev) - 0.5 * pown((value - mean) * inv_stddev, 2) - log(erf((high - mean) * M_SQRT1_2 * inv_stddev) - erf((low - mean) * M_SQRT1_2 * inv_stddev));
}
"""  # TODO: implement

power_law_templates = """
cdouble power_law(const cdouble value, const cdouble slope, const cdouble low, const cdouble high) {
    if(slope == -1.) {
        return 1. / (value * (log(high) - log(low)));
    }
    return pow(value, slope) * (1.+slope) / (pow(high, 1.+slope) - pow(low, 1.+slope));
}

cdouble power_law_falling(const cdouble value, const cdouble slope, const cdouble cutoff) {
    return pow(value, slope) * (-1.-slope) * pow(cutoff, -1.-slope);
}

cdouble log_power_law(const cdouble value, const cdouble slope, const cdouble low, const cdouble high) {
    if(slope == -1.) {
        return -log(value - log(log(high) - log(low)));
    }
    else if(slope < -1.) {
        return log(-1.-slope) + slope * log(value) - log(pow(low, 1.+slope) - pow(high, 1.+slope));
    }
    else {
        return log(1.+slope) + slope * log(value) - log(pow(high, 1.+slope) - pow(low, 1.+slope));
    }

}

cdouble log_power_law_falling(const cdouble value, const cdouble slope, const cdouble cutoff) {
    return log(-1.-slope) + slope * log(value) - (1.+slope) * log(cutoff);
}
"""


def basic_code(queue):
    if cdouble(queue)(42).nbytes >= 8:
        type_source = '#define cdouble double'
    else:
        print('WARNING: no 64bit float support available for this device.')
        type_source = '#define cdouble float'
    return '\n'.join([type_source, math_constants, sum_template,
                      product_template, logsumexp_template, logaddexp_template,
                      mean_template, stddev_template, min_template,
                      max_template, gaussian_pdf_templates,
                      trunc_gaussian_pdf_templates, power_law_templates])
