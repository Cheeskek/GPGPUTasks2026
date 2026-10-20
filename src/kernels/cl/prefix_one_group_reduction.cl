#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

#define BUF_SIZE (GROUP_SIZE / WARP_SIZE)

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_one_group_reduction(
    __global uint* pow2_sum,
    unsigned int from,
    unsigned int to)
{
    __local unsigned int buf[BUF_SIZE];
    const unsigned int lid = get_local_id(0);

    unsigned int x = get_global_id(0);
    unsigned int sum = sub_group_reduce_add(pow2_sum[x]);
    x >>= WARP_SHIFT;
    if ((lid & (WARP_SIZE - 1)) == 0) {
        pow2_sum[to + x] = buf[lid >> WARP_SHIFT] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int sum = buf[0];
        for (int i = 1; i < BUF_SIZE; ++i) {
            sum += buf[i];
            buf[i] = sum;
        }
    }
}
