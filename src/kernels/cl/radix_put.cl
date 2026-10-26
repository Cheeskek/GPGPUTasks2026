#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define IDX_BAD 0xFFFFFFFFu

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
    __local unsigned int idxs[GROUP_SIZE];
    __local unsigned int values[GROUP_SIZE];

    unsigned int x = get_global_id(0);
    unsigned int lid = get_local_id(0);

    unsigned int marker = (input[x] >> (iter * RADIX_SHIFT)) & ((1 << RADIX_SHIFT) - 1);

    unsigned int loc = prefix_locs[x];
    unsigned int start = start_loc[0];

    if (marker == mask) {
        idxs[lid] = loc + start - 1;
        values[lid] = input[x];
    } else {
        idxs[lid] = IDX_BAD;
        values[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 2; k <= GROUP_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = lid ^ j;

            if (partner > lid) {
                unsigned f = idxs[lid], s = idxs[partner];

                int ascending = ((lid & k) == 0);

                if ((ascending && f > s) || (!ascending && f < s)) {
                    idxs[lid] = s;
                    idxs[partner] = f;
                    f = values[lid], s = values[partner];
                    values[lid] = s;
                    values[partner] = f;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (idxs[lid] != IDX_BAD) {
        output[idxs[lid]] = values[lid];
    }

    if (x == n - 1) {
        if (start + loc >= n) {
            start_loc[0] = 0;
        } else {
            start_loc[0] += loc;
        }
    }
}
