#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_final_reduction(
    __global uint* pow2_sum,
    unsigned int from,
    unsigned int to)
{
    unsigned int x = get_global_id(0);
    unsigned int num = to - from;

    __local unsigned int buf[GROUP_SIZE * 2];

    pow2_sum[to + x] = buf[x] = pow2_sum[from + 2 * x] + pow2_sum[from + 2 * x + 1];


    num /= 2;
    from = to;
    to = from + num;

    unsigned int buf_start = GROUP_SIZE;
    unsigned int buf_end = buf_start + num;

    barrier(CLK_LOCAL_MEM_FENCE);

    while (num > 1) {
        if (x < num) {
            pow2_sum[to + x] = buf[buf_end + x] = buf[buf_start + 2 * x] + buf[buf_start + 2 * x + 1];
        }
    }
}
