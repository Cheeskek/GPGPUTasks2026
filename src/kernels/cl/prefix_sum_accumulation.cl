#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_accumulation(
    __global const uint* pow2_sum,
    __global       uint* prefix_sum_accum,
    __global const uint* group_sum_accum,
    unsigned int n)
{
    unsigned int group = get_global_id(0) >> GROUP_SHIFT;
    unsigned int group_v = group_sum_accum[group];
    unsigned int x = get_global_id(0) & (((1 << GROUP_SHIFT) - 1) & ~(WARP_SIZE - 1));
    unsigned int added = group << GROUP_SHIFT;
    unsigned int from = 0;
    unsigned int to = n;
    unsigned int sum = 0;

#if USE_SUBGROUPS
    sum = sub_group_scan_inclusive_add(pow2_sum[get_global_id(0)]);
#else
    __local unsigned int cache[GROUP_SIZE];
    const unsigned int lid = get_local_id(0);
    const unsigned int lane = lid & (WARP_SIZE - 1);
    const unsigned int warp = lid >> WARP_SHIFT;

    cache[lid] = pow2_sum[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (unsigned int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        unsigned int v = 0;
        if (lane >= offset) {
            v = cache[warp * WARP_SIZE + lane - offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane >= offset) {
            cache[warp * WARP_SIZE + lane] += v;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum = cache[lid];
#endif

    from = to;
    n >>= WARP_SHIFT;
    to = from + n;
    x >>= WARP_SHIFT;
    added >>= WARP_SHIFT;

    #pragma unroll
    for (int i = WARP_SHIFT; i < GROUP_SHIFT; ++i) {
        if (x & 1) {
            sum += pow2_sum[from + x + added - 1];
        }
        x >>= 1;
        added >>= 1;
        from = to;
        n /= 2;
        to += n;
    }
    prefix_sum_accum[get_global_id(0)] = sum + group_v;
}
