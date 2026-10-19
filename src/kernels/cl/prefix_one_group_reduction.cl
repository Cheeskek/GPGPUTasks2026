#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_one_group_reduction(
    __global uint* pow2_sum,
    unsigned int from,
    unsigned int to,
    unsigned int limit)
{
    __local unsigned int buf[GROUP_SIZE];
    unsigned int start = from;
    unsigned int num = to - from;
    if (get_global_id(0) < num) {
        unsigned int x = start + get_global_id(0);
        rassert(x - start < GROUP_SIZE, 510987594);
        buf[x - start] = pow2_sum[x];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    while (from < limit) {
        if (2 * get_global_id(0) < num) {
            unsigned int x = from + 2 * get_global_id(0);
            rassert(to + get_global_id(0) - start < GROUP_SIZE, 510987595);
            buf[to + get_global_id(0) - start] = buf[x - start] + buf[x + 1 - start];
        }
        from = to;
        num = (num + 1) / 2;
        to += num;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    unsigned int x = start + get_global_id(0);
    rassert(x - start < GROUP_SIZE, 510987606);
    pow2_sum[x] = buf[x - start];
}
