#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE_X, GROUP_SIZE_Y, 1)))
__kernel void aplusb_matrix_good(__global const uint* a,
                     __global const uint* b,
                     __global       uint* c,
                     unsigned int width,
                     unsigned int height)
{
    const unsigned int idx = get_global_id(0) + get_global_id(1) * width;
    for (int y = 0; y < height; y += 1) {
        for (int x = idx; x < width; x += GROUP_SIZE) {
            c[y * width + x] = a[y * width + x] + b[y * width + x];
        }
    }
}
