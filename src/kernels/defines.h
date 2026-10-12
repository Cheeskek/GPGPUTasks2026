#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   256
#define GROUP_SIZE_X 16
#define GROUP_SIZE_Y 16

#define COOP_GROUP_SIZE_X (GROUP_SIZE_X * 2)
#define COOP_GROUP_SIZE_Y (GROUP_SIZE_Y / 2)

#define COOPMAT_TILE_W 128
#define COOPMAT_TILE_H 128
#define COOPMAT_TILE_K 32
#define COOPMAT_WG_W   4
#define COOPMAT_WG_H   2

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#endif // pragma once
