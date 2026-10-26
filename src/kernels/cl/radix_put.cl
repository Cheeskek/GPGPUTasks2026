#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_put(
    __global uint* input,
    __global uint* output,
    __global uint* prefix_locs,
    __global uint* start_loc, // one value, not array
    unsigned int n,
    unsigned int iter,
    unsigned int mask)
{
    unsigned int x = get_global_id(0);
    unsigned int marker = (input[x] >> (iter * RADIX_SHIFT)) & ((1 << RADIX_SHIFT) - 1);
    if (marker == mask) {
        unsigned int loc = prefix_locs[x];
        unsigned int start = start_loc[0];
        output[start + loc - 1] = input[x];
    }

    if (x == n - 1) {
        unsigned int loc = prefix_locs[x];
        unsigned int start = start_loc[0];
        if (start + loc >= n) {
            start_loc[0] = 0;
        } else {
            start_loc[0] += loc;
        }
    }
}
