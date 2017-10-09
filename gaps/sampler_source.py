#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source code for the OpenCL implementation of the affine-invariant
ensample-sampler.

@author: Sebastian M. Gaebel
@email: sebastian.gaebel@ligo.org
"""


ensemble_sampler_source = """
size_t random_index(const cfloat random_number, const size_t idx) {
    size_t index = random_number * (GROUP_SIZE-1);
    if(index >= idx) {
        index++;
    }
    return index;
}


cfloat random_stretch(const cfloat random_number) {
    const cfloat temp = random_number * (SCALE_PARAMETER-1.) + 1.;
    return temp*temp / (cfloat)SCALE_PARAMETER;
}


void stretch_math(__local cfloat * prev_pos, __local cfloat * random_pos,
                  const cfloat stretch_factor, cfloat * dst) {
    for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
        dst[dim_idx] = random_pos[dim_idx] - stretch_factor * (random_pos[dim_idx] - prev_pos[dim_idx]);
    }
}


__kernel void ensemble_sampling(__global const cfloat initial_parameters[N_WALKERS][N_DIM],
                                __global const cfloat uniform_random[N_WALKERS][N_SAMPLES][3],
                                __global cfloat samples[N_WALKERS][N_SAMPLES][N_DIM],
                                __global cfloat logP_values[N_WALKERS][N_SAMPLES],
                                __global size_t acceptance_counter[N_WALKERS]
                                GAPS_FUNC_DEF
                                #ifdef DEBUG_OUTPUT
                                // This looks weird, but otherwise we would have to add the
                                // comma from python.
                                , __global cfloat debug_values[N_WALKERS][N_SAMPLES]
                                , __global cfloat proposals[N_WALKERS][N_SAMPLES][N_DIM]
                                #endif
                                ) {
    // Setup constants
    const size_t global_idx = get_global_id(0);
    const size_t local_idx = get_local_id(0);

    // Buffers and convenience objects
    __local cfloat prev_positions[GROUP_SIZE][N_DIM];
    __local cfloat next_positions[GROUP_SIZE][N_DIM];
    __local cfloat prev_logP[GROUP_SIZE];
    __local cfloat next_logP[GROUP_SIZE];
    cfloat trial_parameters[N_DIM];
    cfloat trial_logP;
    size_t accepted = 0;

    // Initialize the walker
    for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
        samples[global_idx][0][dim_idx] = initial_parameters[global_idx][dim_idx];
        prev_positions[local_idx][dim_idx] = initial_parameters[global_idx][dim_idx];
        trial_parameters[dim_idx] = initial_parameters[global_idx][dim_idx];
    }
    // trial_logP is used solely as buffer to not duplicate function calls
    #ifdef INITIAL_LOGP
    trial_logP = given_logP[global_idx];
    #else
    trial_logP = logP_fn(trial_parameters GAPS_FUNC_CALL);
    #endif
    prev_logP[local_idx] = trial_logP;
    logP_values[global_idx][0] = trial_logP;

    #ifdef DEBUG_OUTPUT
    debug_values[global_idx][0] = debug_fn(data_samples GAPS_FUNC_CALL);
    for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
        proposals[global_idx][0][dim_idx] = initial_parameters[global_idx][dim_idx];
    }
    #endif

    // BLOCK to sync local buffers
    barrier(CLK_LOCAL_MEM_FENCE);

    for(size_t sample_idx = 1; sample_idx < N_SAMPLES; sample_idx++) {
        // Random index unequal to the current local idx
        const size_t random_idx = random_index(uniform_random[global_idx][sample_idx][0], local_idx);
        // Random stretch value
        const cfloat stretch = random_stretch(uniform_random[global_idx][sample_idx][1]);
        // Acceptance probability
        const cfloat acceptance_random = uniform_random[global_idx][sample_idx][2];

        // Create a trial parameter value
        stretch_math(prev_positions[local_idx], prev_positions[random_idx], stretch, trial_parameters);
        trial_logP = logP_fn(trial_parameters GAPS_FUNC_CALL);
        #ifdef DEBUG_OUTPUT
        debug_values[global_idx][sample_idx] = debug_fn(trial_parameters GAPS_FUNC_CALL);
        for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
            proposals[global_idx][sample_idx][dim_idx] = trial_parameters[dim_idx];
        }
        #endif

        // Check for acceptance and write the correct value into the 'next' buffer
        const cfloat delta_logP = (N_DIM-1) * log(stretch) + trial_logP - prev_logP[local_idx];
        if(delta_logP > log(acceptance_random)) {
            for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
                next_positions[local_idx][dim_idx] = trial_parameters[dim_idx];
            }
            next_logP[local_idx] = trial_logP;
            accepted++;
        }
        else {
            for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
                next_positions[local_idx][dim_idx] = prev_positions[local_idx][dim_idx];
            }
            next_logP[local_idx] = prev_logP[local_idx];
        }

        // BLOCK to sync local buffers
        barrier(CLK_LOCAL_MEM_FENCE);

        // Copy next buffer into global and prev
        for(size_t dim_idx = 0; dim_idx < N_DIM; dim_idx++) {
            prev_positions[local_idx][dim_idx] = next_positions[local_idx][dim_idx];
            samples[global_idx][sample_idx][dim_idx] = prev_positions[local_idx][dim_idx];
        }
        prev_logP[local_idx] = next_logP[local_idx];
        logP_values[global_idx][sample_idx] = prev_logP[local_idx];

        // BLOCK to sync local buffers
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    acceptance_counter[global_idx] = accepted;
    return;
}
"""
