#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_reduction(
    __global uint* pow2_sum,
    unsigned int from,
    unsigned int to)
{
    unsigned int x = get_global_id(0);
    if (x < to - from && !(x & 1)) {
        pow2_sum[to + (x / 2)] = pow2_sum[from + x] + pow2_sum[from + x + 1];
    }
}
