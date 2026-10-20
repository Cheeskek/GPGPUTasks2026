#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_ACC, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    __global       uint* group_sum_accum,
    unsigned int n)
{
    unsigned int group = get_global_id(0) >> ACC_SHIFT;
    unsigned int group_v = group_sum_accum[group];
    unsigned int x = get_global_id(0) & (((1 << ACC_SHIFT) - 1) & ~(WARP_SIZE - 1));
    unsigned int added = group << ACC_SHIFT;
    unsigned int from = 0;
    unsigned int to = n;
    unsigned int init_sum = sub_group_scan_inclusive_add(pow2_sum[get_global_id(0)]);
    unsigned int sum = init_sum;

    from = to;
    n >>= WARP_SHIFT;
    to = from + n;
    x >>= WARP_SHIFT;
    added >>= WARP_SHIFT;

    #pragma unroll
    for (int i = WARP_SHIFT; i < ACC_SHIFT; ++i) {
        if (x & 1) {
            sum += pow2_sum[from + x + added - 1];
        }
        x >>= 1;
        added >>= 1;
        from = to;
        n = (n + 1) / 2;
        to += n;
    }
    prefix_sum_accum[get_global_id(0)] = sum + group_v;
}
