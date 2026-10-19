#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    unsigned int n)
{
    unsigned int x = get_global_id(0) + 1;
    unsigned int from = 0;
    unsigned int to = n;
    unsigned int sum = 0;
    while (x > 0) {
        if (x & 1) {
            /* rassert(from + x - 1 < 2 * get_global_size(0), 198744); */
            sum += pow2_sum[from + x - 1];
        }
        x /= 2;
        from = to;
        n = (n + 1) / 2;
        to += n;
    }
    prefix_sum_accum[get_global_id(0)] = sum;
}
