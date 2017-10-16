# GAPS - GPU Accelerated Parallel Sampler

GAPS is an affine invariant ensemble sampler by Goodman & Weare ([DOI: 10.2140/camcos.2010.5.65](https://msp.org/camcos/2010/5-1/p04.xhtml)). The implementation is largely based on [emcee](http://dfm.io/emcee/) by Dan Foreman-Mackey.

### Installation:

Since the code is still in early stages the you may want to install this module as 'editable' via:

```
git clone https://github.com/sgaebel/GAPS /path/to/GAPS/
pip install -e /path/to/GAPS/
```

The main dependencies are `numpy`, `matplotlib`, and `pyopencl`. Installation instructions for `pyopencl` are available in its [documentation](https://documen.tician.de/pyopencl/misc.html) and may vary strongly based on the operating system.

Under Windows a good method is to install the [AMD APP SDK](http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/), and the binaries for [cffi](http://www.lfd.uci.edu/~gohlke/pythonlibs/#cffi) and [pyopencl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl).

### Usage:

The core component of this module is the `gaps.run_sampler()` function, which requires the user to define a `logP_fn` function. This function signature is required to be of the form `cfloat logP_fn(const cfloat parameters USER_DATA)`, where `USER_DATA` enables the user the pass one or more global arrays through the sampler to `logP_fn`. Currently `USER_DATA` arrays may only be of type `__global const cdouble`. To get started quickly, check out `examples/fitting_2d_gaussian.py`.

### Examples:

 * `cfloat logP_fn(const cfloat parameters)`
 * `cfloat logP_fn(const cfloat parameters, __global const cdouble some_data[DIM1])`
 * `cfloat logP_fn(const cfloat parameters, __global const cdouble some_data[DIM1][1200], __global const cdouble other_data[DIM2])`

In addition to the sampler itself, GAPS also provides a number of mathematical constants, functions, and distributions:

 * `M_1_SQRTPI : 1 / sqrt(pi)`
 * `M_1_SQRT2PI : 1 / sqrt(2 * pi)`
 * `M_LOG_1_SQRTPI : log(1 / sqrt(pi))`
 * `M_LOG_1_SQRT2PI : log(1 / sqrt(2 * pi))`
 * `M_SQRT_2_PI : sqrt(2 * pi)`
 * `M_LOG_SQRT_2_PI : log(sqrt(2 * pi))`
 * `M_1_SQRT2 : 1 / sqrt(2)`
 * `cdouble sum(__global const cdouble * iterable, const size_t length)`
 * `cdouble product(__global const cdouble * iterable, const size_t length)`
 * `cdouble logsumexp(__global const cdouble * log_values, size_t length)`
 * `cdouble logaddexp(const cdouble x, const cdouble y)`
 * `cdouble mean(__global const cdouble * iterable, const size_t length)`
 * `cdouble stddev(__global const cdouble * iterable, const size_t length)`
 * `cdouble iter_min(__global const cdouble * iterable, const size_t length)`
 * `cdouble iter_max(__global const cdouble * iterable, const size_t length)`
 * `cdouble gaussian(const cdouble value, const cdouble mean, const cdouble stddev)`
 * `cdouble log_gaussian(const cdouble value, const cdouble mean, const cdouble stddev)`
 * `cdouble trunc_gaussian(const cdouble value, const cdouble mean, const cdouble stddev, const cdouble low, const cdouble high)`
 * `cdouble log_trunc_gaussian(const cdouble value, const cdouble mean, const cdouble stddev, const cdouble low, const cdouble high)`
 * `cdouble power_law(const cdouble value, const cdouble slope, const cdouble low, const cdouble high)`
 * `cdouble power_law_falling(const cdouble value, const cdouble slope, const cdouble cutoff)`
 * `cdouble log_power_law(const cdouble value, const cdouble slope, const cdouble low, const cdouble high)`
 * `cdouble log_power_law_falling(const cdouble value, const cdouble slope, const cdouble cutoff)`

Currently there is no templating available, so all function only accept `__global` arrays. These functions are defined in `gaps/auxiliary_sources.py` and can be copied and/or modified if needed.

Please refer to the OpenCL documentations for more basic functions and constants:

 * [Math functions](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/mathFunctions.html)
 * [Math constants](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/mathConstants.html)

Some examples are available in `examples/` and illustrate basic use cases.
