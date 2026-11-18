// D20 doc check: which WMMA builtins compile for gfx1151 (compile only).
#include <hip/hip_runtime.h>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef int i4v __attribute__((ext_vector_type(4)));
typedef int i2v __attribute__((ext_vector_type(2)));
typedef int i8v __attribute__((ext_vector_type(8)));
__global__ void k_f16(f8 *o, const h16 *a, const h16 *b) { o[threadIdx.x] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[threadIdx.x], b[threadIdx.x], o[threadIdx.x]); }
__global__ void k_bf16(f8 *o, const s16 *a, const s16 *b) { o[threadIdx.x] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[threadIdx.x], b[threadIdx.x], o[threadIdx.x]); }
__global__ void k_iu8(i8v *o, const i4v *a, const i4v *b) { o[threadIdx.x] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a[threadIdx.x], true, b[threadIdx.x], o[threadIdx.x], false); }
__global__ void k_iu4(i8v *o, const i2v *a, const i2v *b) { o[threadIdx.x] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a[threadIdx.x], true, b[threadIdx.x], o[threadIdx.x], false); }
