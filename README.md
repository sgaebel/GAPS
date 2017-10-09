    GAPS - GPU Accelerated Parallel Sampler

The MCMC sampler implemented here is a affine invariant ensamble sampler
by Goodman & Weare (DOI: 10.2140/camcos.2010.5.65). The implementation
is largely based on emcee (http://dfm.io/emcee/) by Dan Foreman-Mackey.

The core component is the 'run_sampler' function, which requires the
user to define a logP_fn function. This function signature is required
to be of the form 'cfloat logP_fn(const cfloat parameters USER_DATA)',
where USER_DATA enables the user the pass global array through the 
sampler to logP_fn. USER_DATA may, it this point, only be of type 
'__global const cdouble'.
Examples:
    'cfloat logP_fn(const cfloat parameters)'
    'cfloat logP_fn(const cfloat parameters,
                    __global const cdouble some_data[DIM1])'
    'cfloat logP_fn(const cfloat parameters,
                    __global const cdouble some_data[DIM1][DIM2],
                    __global const cdouble other_data[DIM3])'
In addition to the sampler itself, GAPS also make a number of
mathematical constants, functions, and distributions avilable:
    M_1_SQRTPI : 1 / sqrt(pi)
    M_1_SQRT2PI : 1 / sqrt(2 * pi)
    M_LOG_1_SQRTPI : log(1 / sqrt(pi))
    M_LOG_1_SQRT2PI : log(1 / sqrt(2 * pi))
    M_SQRT_2_PI : sqrt(2 * pi)
    M_LOG_SQRT_2_PI : log(sqrt(2 * pi))
    M_1_SQRT2 : 1 / sqrt(2)
    cdouble sum(__global const cdouble * iterable, const size_t length)
    cdouble product(__global const cdouble * iterable, const size_t length)
    cdouble logsumexp(__global const cdouble * log_values, size_t length)
    cdouble logaddexp(const cdouble x, const cdouble y)
    cdouble mean(__global const cdouble * iterable, const size_t length)
    cdouble stddev(__global const cdouble * iterable, const size_t length)
    cdouble iter_min(__global const cdouble * iterable, const size_t length)
    cdouble iter_max(__global const cdouble * iterable, const size_t length)
    cdouble gaussian(const cdouble value, const cdouble mean, const cdouble stddev)
    cdouble log_gaussian(const cdouble value, const cdouble mean, const cdouble stddev)
    cdouble trunc_gaussian(const cdouble value, const cdouble mean, const cdouble stddev,
                           const cdouble low, const cdouble high)
    cdouble log_trunc_gaussian(const cdouble value, const cdouble mean, const cdouble stddev,
                          const cdouble low, const cdouble high)
	cdouble power_law(const cdouble value, const cdouble slope, const cdouble low, const cdouble high)
	cdouble power_law_falling(const cdouble value, const cdouble slope, const cdouble cutoff)
	cdouble log_power_law(const cdouble value, const cdouble slope, const cdouble low, const cdouble high)
	cdouble log_power_law_falling(const cdouble value, const cdouble slope, const cdouble cutoff)
Currently there is no templating available, so all function only accept
__global arrays. The functions are available in gaps/auxiliary_sources.py
to be copied and/or modified if necessary.

Some simple examples are available in examples/ and illustrate the basic
use cases.
