// block_sparse_blk128_sm100a.cu -- the 128-token-block instantiation of the torch binding.
//
// Same source as block_sparse_sm100a.cu with VSA_BLK128 set: the kernel and launch land in
// namespace vsa_blk128 (distinct symbols, no ODR clash with the blk64 objects) and the
// exported entry point becomes block_sparse_sm100a_blk128_fwd.
#define VSA_BLK128 true
#include "block_sparse_sm100a.cu"
