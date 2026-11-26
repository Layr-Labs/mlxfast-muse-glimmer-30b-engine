// Copyright © 2025-2026 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/fp4.h"
#include "mlx/backend/metal/kernels/fp8.h"

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8, int bits>
inline constexpr short get_fp_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    // Use nv scale
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}

template <int bits>
struct Quantize {
  uint8_t operator()(float x) thread {
    if (bits == 8) {
      return fp8_e4m3(x).bits;
    } else {
      return fp4_e2m1(x).bits;
    }
  }
};

template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename U, int bits>
inline void dequantize(uint8_t w, float scale, threadgroup U* w_local) {
  if constexpr (bits == 4) {
    w_local[0] = static_cast<U>(scale * Dequantize<4, float>{}(w));
    w_local[1] = static_cast<U>(scale * Dequantize<4, float>{}(w >> 4));
  } else {
    w_local[0] = static_cast<U>(scale * Dequantize<8, float>{}(w));
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    bool has_global_scale = false>
struct FPQuantizedNAXBlockLoader {
  MLX_MTL_CONST short pack_factor = get_fp_pack_factor<8, bits>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;

  MLX_MTL_CONST short n_reads_per_scale = (n_reads * pack_factor) <= group_size
      ? n_reads
      : (group_size / pack_factor);
  MLX_MTL_CONST short n_steps_per_read = n_reads / n_reads_per_scale;

  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;
  // nvfp4 tensor scale, folded into the group scale as fp_dequantize does.
  // Kept in float: it is ~1e-5, so in fp16 small scales lose most bits.
  float inv_scale_enc = 1.0f;

  FPQuantizedNAXBlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const device float* global_scale = nullptr) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id) {
    if constexpr (has_global_scale) {
      inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);
    }
  }

  void fetch_nvfp4(thread uint2* q_buf, thread uint8_t* s_buf) const thread {
    if constexpr (n_steps_per_read == 2) {
      const uint4 q4 = *(const device uint4*)src;
      q_buf[0] = q4.xy;
      q_buf[1] = q4.zw;
      const uchar2 s2 = *(const device uchar2*)scales;
      s_buf[0] = s2.x;
      s_buf[1] = s2.y;
    } else {
      const device uint2* src_u2 = (const device uint2*)src;
      for (int i = 0; i < n_steps_per_read; i++) {
        q_buf[i] = src_u2[i];
        s_buf[i] = scales[i];
      }
    }
  }

  void store_nvfp4(
      threadgroup T* ws_base,
      const thread uint2* q_buf,
      const thread uint8_t* s_buf) const thread {
    threadgroup vec<T, 8>* dst_v8 =
        (threadgroup vec<T, 8>*)(ws_base + bi * dst_ld + bj * pack_factor);
    for (int i = 0; i < n_steps_per_read; i++) {
      const half scale = dequantize_scale<half, 16>(s_buf[i]);
      const uint2 q = q_buf[i];
      {
        const uint lo0 = (q.x & 0x000f) | ((q.x & 0x00f0) << 12);
        const uint hi0 = ((q.x >> 8) & 0x000f) | ((q.x & 0xf000) << 4);
        half2 a0 = (as_type<half2>(
                        ((lo0 & 0x00070007) << 9) |
                        ((lo0 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;
        half2 b0 = (as_type<half2>(
                        ((hi0 & 0x00070007) << 9) |
                        ((hi0 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;

        const uint qx1 = q.x >> 16;
        const uint lo1 = (qx1 & 0x000f) | ((qx1 & 0x00f0) << 12);
        const uint hi1 = ((qx1 >> 8) & 0x000f) | ((qx1 & 0xf000) << 4);
        half2 a1 = (as_type<half2>(
                        ((lo1 & 0x00070007) << 9) |
                        ((lo1 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;
        half2 b1 = (as_type<half2>(
                        ((hi1 & 0x00070007) << 9) |
                        ((hi1 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;
        dst_v8[2 * i + 0] = vec<T, 8>(
            T(a0.x), T(a0.y), T(b0.x), T(b0.y), T(a1.x), T(a1.y), T(b1.x),
            T(b1.y));
      }
      {
        const uint lo2 = (q.y & 0x000f) | ((q.y & 0x00f0) << 12);
        const uint hi2 = ((q.y >> 8) & 0x000f) | ((q.y & 0xf000) << 4);
        half2 a2 = (as_type<half2>(
                        ((lo2 & 0x00070007) << 9) |
                        ((lo2 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;
        half2 b2 = (as_type<half2>(
                        ((hi2 & 0x00070007) << 9) |
                        ((hi2 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;

        const uint qy1 = q.y >> 16;
        const uint lo3 = (qy1 & 0x000f) | ((qy1 & 0x00f0) << 12);
        const uint hi3 = ((qy1 >> 8) & 0x000f) | ((qy1 & 0xf000) << 4);
        half2 a3 = (as_type<half2>(
                        ((lo3 & 0x00070007) << 9) |
                        ((lo3 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;
        half2 b3 = (as_type<half2>(
                        ((hi3 & 0x00070007) << 9) |
                        ((hi3 & 0x00080008) << 12)) *
                    half2(16384.0h)) *
            scale;
        dst_v8[2 * i + 1] = vec<T, 8>(
            T(a2.x), T(a2.y), T(b2.x), T(b2.y), T(a3.x), T(a3.y), T(b3.x),
            T(b3.y));
      }
    }
  }

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if constexpr (
        bits == 4 && group_size == 16 && !has_global_scale &&
        n_reads_per_scale == 8) {
      uint2 q_buf[n_steps_per_read];
      uint8_t s_buf[n_steps_per_read];
      fetch_nvfp4(q_buf, s_buf);
      store_nvfp4(dst - (bi * dst_ld + bj * pack_factor), q_buf, s_buf);
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      float scale =
          float(dequantize_scale<T, group_size>(scales[i])) * inv_scale_enc;
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize<T, bits>(
            src[k * bytes_per_pack], scale, dst + k * pack_factor);
        k++;
      }
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      const bool scale_in_bounds = (bj + k) * pack_factor < src_tile_dim.x;
      float scale = scale_in_bounds
          ? float(dequantize_scale<T, group_size>(scales[i])) * inv_scale_enc
          : 0.0f;
      for (int j = 0; j < n_reads_per_scale; j++) {
        if ((bj + k) * pack_factor < src_tile_dim.x) {
          dequantize<T, bits>(
              src[k * bytes_per_pack], scale, dst + k * pack_factor);
        } else {
          for (int p = 0; p < pack_factor; p++) {
            dst[k * pack_factor + p] = T(0);
          }
        }
        k++;
      }
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      scales += n_groups;
    } else {
      scales += group_stride;
    }
  }
};

using namespace mlx::steel;

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool has_global_scale = false,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat,
    const bool double_buffer_ws = false>
METAL_FUNC void fp_qmm_t_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    threadgroup Wtype* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_fp_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  // Instantiate Loader
  using loader_w_t = FPQuantizedNAXBlockLoader<
      Wtype,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      has_global_scale>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid, global_scale);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      if constexpr (
          kAlignedN.value && bits == 4 && group_size == 16 &&
          !has_global_scale && loader_w_t::n_reads_per_scale == 8 &&
          loader_w_t::BCOLS_PACKED * BN >= WM * WN * SIMD_SIZE) {
        constexpr int ws_tile_size = BN * BK_padded;
        uint2 q_buf[loader_w_t::n_steps_per_read];
        uint8_t s_buf[loader_w_t::n_steps_per_read];
        loader_w.fetch_nvfp4(q_buf, s_buf);
        loader_w.next();
        if constexpr (double_buffer_ws) {
          loader_w.store_nvfp4(Ws, q_buf, s_buf);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          int stage = 0;
          for (int k = 0; k < K; k += BK) {
            if (k + BK < K) {
              loader_w.fetch_nvfp4(q_buf, s_buf);
              loader_w.next();
            }

            threadgroup Wtype* Ws_cur = Ws + stage * ws_tile_size;
            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<Wtype, TN, TK> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(x + kk1, K);
              } else {
                Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
              }

              Btile.template load<Wtype, BK_padded, 1>(
                  Ws_cur + tn * BK_padded + kk1);

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<transpose_a>{},
                  Btile,
                  metal::bool_constant<transpose_b>{});

              (void)compiler_barrier;
            }

            if (k + BK < K) {
              loader_w.store_nvfp4(
                  Ws + (1 - stage) * ws_tile_size, q_buf, s_buf);
              threadgroup_barrier(mem_flags::mem_threadgroup);
              stage = 1 - stage;
            }

            x += BK;
          }
        } else {
          for (int k = 0; k < K; k += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_w.store_nvfp4(Ws, q_buf, s_buf);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (k + BK < K) {
              loader_w.fetch_nvfp4(q_buf, s_buf);
              loader_w.next();
            }

            STEEL_PRAGMA_NO_UNROLL
            for (int kk1 = 0; kk1 < BK; kk1 += SK) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<Wtype, TN, TK> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(x + kk1, K);
              } else {
                Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
              }

              Btile.template load<Wtype, BK_padded, 1>(
                  Ws + tn * BK_padded + kk1);

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<transpose_a>{},
                  Btile,
                  metal::bool_constant<transpose_b>{});

              (void)compiler_barrier;
            }

            x += BK;
          }
        }
      } else {
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(short2(BK, tgp_bn));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<Wtype, TN, TK> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(x + kk1, K);
            } else {
              Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
            }

            Btile.template load<Wtype, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<transpose_a>{},
                Btile,
                metal::bool_constant<transpose_b>{});

            (void)compiler_barrier;
          }

          x += BK;
          loader_w.next();
        }
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <typename Wtype, int group_size, int bits>
METAL_FUNC vec<Wtype, 4> dequantize_fp_x4(uint q, uint8_t s) {
  half2 a;
  half2 b;
  if constexpr (bits == 4) {
    const uint lo = (q & 0x000f) | ((q & 0x00f0) << 12);
    const uint hi = ((q >> 8) & 0x000f) | ((q & 0xf000) << 4);
    a = as_type<half2>(((lo & 0x00070007) << 9) | ((lo & 0x00080008) << 12));
    b = as_type<half2>(((hi & 0x00070007) << 9) | ((hi & 0x00080008) << 12));
  } else {
    const uint lo = (q & 0x000000ff) | ((q & 0x0000ff00) << 8);
    const uint hi = ((q >> 16) & 0x000000ff) | ((q >> 8) & 0x00ff0000);
    a = as_type<half2>(((lo & 0x00800080) << 8) | ((lo & 0x007f007f) << 7));
    b = as_type<half2>(((hi & 0x00800080) << 8) | ((hi & 0x007f007f) << 7));
  }
  if constexpr (group_size == 16) {
    const half scale = dequantize_scale<half, group_size>(s);
    a = (a * half2(16384.0h)) * scale;
    b = (b * half2(16384.0h)) * scale;
    return vec<Wtype, 4>(Wtype(a.x), Wtype(a.y), Wtype(b.x), Wtype(b.y));
  } else {
    const float scale = dequantize_scale<float, group_size>(s) *
        (bits == 4 ? 16384.0f : 256.0f);
    return vec<Wtype, 4>(
        Wtype(float(a.x) * scale),
        Wtype(float(a.y) * scale),
        Wtype(float(b.x) * scale),
        Wtype(float(b.y) * scale));
  }
}

template <typename Wtype>
METAL_FUNC vec<Wtype, 4> dequantize_fp4_x4_scaled(uint q, half scale) {
  const uint lo = (q & 0x000f) | ((q & 0x00f0) << 12);
  const uint hi = ((q >> 8) & 0x000f) | ((q & 0xf000) << 4);
  half2 a = (as_type<half2>(
                 ((lo & 0x00070007) << 9) | ((lo & 0x00080008) << 12)) *
             half2(16384.0h)) *
      scale;
  half2 b = (as_type<half2>(
                 ((hi & 0x00070007) << 9) | ((hi & 0x00080008) << 12)) *
             half2(16384.0h)) *
      scale;
  return vec<Wtype, 4>(Wtype(a.x), Wtype(a.y), Wtype(b.x), Wtype(b.y));
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool has_global_scale = false,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_n_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;
  (void)M;

  constexpr int pack_factor = get_fp_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = FPQuantizedNAXBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      has_global_scale>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  // const short num_els = min(BM, M - y_row);
  // const short num_outs = min(BN, N - y_col);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid, global_scale);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<Wtype, TK, TN> Btile;

      volatile int compiler_barrier;

      Atile.load(x + kk1, K);
      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  Dtile.store(y + tm * N + tn, N);
}

template <typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  y += tid.z * output_stride;
}

template <bool has_global_scale, typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
    const device float*& global_scale,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  // One global scale per expert, contiguous over the batch dims of w.
  if constexpr (has_global_scale) {
    global_scale += w_idx;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));
  constexpr bool kDoubleBuf =
      (bits == 4 && group_size == 16 && !has_global_scale &&
       sizeof(Wtype) == 2);

  threadgroup Wtype Ws[(kDoubleBuf ? 2 : 1) * BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }
  fp_qmm_t_impl<
      T,
      group_size,
      bits,
      aligned_N,
      has_global_scale,
      BM,
      BK,
      BN,
      WM,
      WN,
      Wtype,
      kDoubleBuf>(
      w, scales, global_scale, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }

  fp_qmm_n_impl<T, group_size, bits, false, BM, BK, BN, WM, WN, Wtype>(
      w, scales, nullptr, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];

  adjust_matrix_offsets<has_global_scale>(
      x,
      w,
      scales,
      global_scale,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_t_impl<
      T,
      group_size,
      bits,
      aligned_N,
      has_global_scale,
      BM,
      BK,
      BN,
      WM,
      WN,
      Wtype>(
      w, scales, global_scale, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<has_global_scale>(
      x,
      w,
      scales,
      global_scale,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_n_impl<T, group_size, bits, false, BM, BK, BN, WM, WN, Wtype>(
      w, scales, nullptr, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    const int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose,
    bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_rhs_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device int32_t* offsets,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    const constant int& num_groups,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_fp_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));
  constexpr int BN_padded = (BN + 16 / sizeof(Wtype));

  using loader_w_t = FPQuantizedNAXBlockLoader<
      Wtype,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      has_global_scale>;

  threadgroup Wtype Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = short(clamp(int(tgp_bm) - tm, 0, int(SM)));
  const short sgp_sn = align_N ? SN : min(int(SN), max(0, (N - (y_col + tn))));

  const bool rows_in_bounds = y_row + tm + SM <= M;
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = x + tm * K;

  // Prepare threadgroup loading operations
  thread loader_w_t loader_w(
      wl + group * stride_w,
      scales + group * stride_s,
      transpose ? K : N,
      Ws,
      simd_group_id,
      simd_lane_id,
      global_scale + group);

  dispatch_bool(rows_in_bounds, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(
              transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<Wtype, BR, BC> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(xn + kk1, K);
          } else {
            Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
          }

          if constexpr (transpose) {
            Btile.template load<Wtype, BK_padded, 1>(Ws + tn * BK_padded + kk1);
          } else {
            Btile.template load<Wtype, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
          }

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Btile,
              metal::bool_constant<transpose>{});

          (void)compiler_barrier;
        }

        xn += BK;
        loader_w.next();
      }

      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_w.load_safe(tile_w);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<Wtype, BR, BC> Btile;

          volatile int compiler_barrier;

          const short psk = min(int(SK), max(0, (k_remain - kk1)));
          Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

          if constexpr (transpose) {
            Btile.template load<Wtype, BK_padded, 1>(Ws + tn * BK_padded + kk1);
          } else {
            Btile.template load<Wtype, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
          }

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<false>{},
              Btile,
              metal::bool_constant<transpose>{});

          (void)compiler_barrier;
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Store results to device memory
      if constexpr (kAlignedN.value) {
        if (sgp_sm == SM) {
          Dtile.store(y + tm * N + tn, N);
        } else {
          Dtile.store_slice(
              y + tm * N + tn, N, short2(0, 0), short2(SN, sgp_sm));
        }
      } else {
        Dtile.store_slice(
            y + tm * N + tn, N, short2(0, 0), short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    int group_size,
    int bits,
    int WK = 8,
    int BN = 32,
    typename Wtype = bfloat>
[[kernel]] void fp_qmv_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  static_assert(BN % 32 == 0, "BN should be divisible by 32");
  static_assert(
      (group_size == 16 && bits == 4) ||
          (group_size == 32 && (bits == 4 || bits == 8)),
      "qmv_nax reads nvfp4, mxfp4 and mxfp8");

  using Frag = BaseNAXFrag;
  using scale_bits_t = metal::conditional_t<group_size == 16, ushort, uint8_t>;
  constexpr int BK = 128;
  constexpr short TK = BK / 16;
  constexpr short TN = BN / 16;
  constexpr int pack_factor = 8 / bits;
  constexpr short loads = bits / 4;

  const short2 coord = Frag::get_coord();
  const short fm = coord.y;
  const int k_lane = 8 * coord.x;
  const int row_bytes = K / pack_factor;
  const int row_scales = K / group_size;

  const int y_col = tid.x * BN;
  auto wl = (const device uint8_t*)w +
      (y_col + fm) * static_cast<int64_t>(row_bytes) + k_lane / pack_factor;
  scales +=
      (y_col + fm) * static_cast<int64_t>(row_scales) + k_lane / group_size;
  x += fm * static_cast<int64_t>(K) + k_lane;
  y += y_col;
  const bool x_rows[2] = {fm < M, fm + 8 < M};

  NAXTile<float, 1, TN> Dtile;
  Dtile.clear();

  if constexpr (bits == 4 && group_size == 16) {
    if (M <= 8) {
      const bool x_row0 = fm < M;
      uint4 ws_cur[TN][2];
      ushort ss_cur[TN][2];
      const int k0 = simd_gid * BK;
      if (k0 < K) {
        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < TN; t++) {
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < 2; r++) {
            const int row = 16 * t + 8 * r;
            ws_cur[t][r] = *(const device uint4*)(
                wl + row * static_cast<int64_t>(row_bytes) + k0 / 2);
            ss_cur[t][r] = *(const device ushort*)(
                scales + row * static_cast<int64_t>(row_scales) + k0 / 16);
          }
        }
      }

      for (int k = k0; k < K; k += WK * BK) {
        vec<T, 8> xs0[4];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 4; i++) {
          xs0[i] = x_row0 ? *(const device vec<T, 8>*)(x + k + 8 * i)
                          : vec<T, 8>(0);
        }

        uint4 ws_next[TN][2];
        ushort ss_next[TN][2];
        const int kn = k + WK * BK;
        if (kn < K) {
          STEEL_PRAGMA_UNROLL
          for (short t = 0; t < TN; t++) {
            STEEL_PRAGMA_UNROLL
            for (short r = 0; r < 2; r++) {
              const int row = 16 * t + 8 * r;
              ws_next[t][r] = *(const device uint4*)(
                  wl + row * static_cast<int64_t>(row_bytes) + kn / 2);
              ss_next[t][r] = *(const device ushort*)(
                  scales + row * static_cast<int64_t>(row_scales) + kn / 16);
            }
          }
        }

        half sc0[TN][2];
        half sc1[TN][2];
        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < TN; t++) {
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < 2; r++) {
            sc0[t][r] = dequantize_scale<half, 16>(uint8_t(ss_cur[t][r]));
            sc1[t][r] = dequantize_scale<half, 16>(uint8_t(ss_cur[t][r] >> 8));
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < TK; f++) {
          typename Frag::template dtype_frag_t<T> a;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++) {
            a[j] = xs0[f / 2][4 * (f % 2) + j];
            a[4 + j] = T(0);
          }
          typename Frag::template dtype_frag_t<Wtype> b[TN];
          STEEL_PRAGMA_UNROLL
          for (short t = 0; t < TN; t++) {
            STEEL_PRAGMA_UNROLL
            for (short r = 0; r < 2; r++) {
              const uint q = ws_cur[t][r][f / 2] >> (16 * (f % 2));
              const half sc = (f < 4) ? sc0[t][r] : sc1[t][r];
              const vec<Wtype, 4> v = dequantize_fp4_x4_scaled<Wtype>(q, sc);
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < 4; j++) {
                b[t][4 * r + j] = v[j];
              }
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short t = 0; t < TN; t += 2) {
            Frag::mma(
                Dtile.frag_at(0, t),
                Dtile.frag_at(0, t + 1),
                a,
                metal::bool_constant<false>{},
                b[t],
                b[t + 1],
                metal::bool_constant<true>{});
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < TN; t++) {
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < 2; r++) {
            ws_cur[t][r] = ws_next[t][r];
            ss_cur[t][r] = ss_next[t][r];
          }
        }
      }
    } else {
      for (int k = simd_gid * BK; k < K; k += WK * BK) {
        vec<T, 8> xs[2][4];
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; r++) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 4; i++) {
            xs[r][i] = x_rows[r]
                ? *(const device vec<T, 8>*)(
                      x + r * 8 * static_cast<int64_t>(K) + k + 8 * i)
                : vec<T, 8>(0);
          }
        }
        uint4 ws[TN][2];
        ushort ss[TN][2];
        half sc0[TN][2];
        half sc1[TN][2];
        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < TN; t++) {
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < 2; r++) {
            const int row = 16 * t + 8 * r;
            ws[t][r] = *(const device uint4*)(
                wl + row * static_cast<int64_t>(row_bytes) + k / 2);
            ss[t][r] = *(const device ushort*)(
                scales + row * static_cast<int64_t>(row_scales) + k / 16);
            sc0[t][r] = dequantize_scale<half, 16>(uint8_t(ss[t][r]));
            sc1[t][r] = dequantize_scale<half, 16>(uint8_t(ss[t][r] >> 8));
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < TK; f++) {
          typename Frag::template dtype_frag_t<T> a;
          STEEL_PRAGMA_UNROLL
          for (short r = 0; r < 2; r++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
              a[4 * r + j] = xs[r][f / 2][4 * (f % 2) + j];
            }
          }
          typename Frag::template dtype_frag_t<Wtype> b[TN];
          STEEL_PRAGMA_UNROLL
          for (short t = 0; t < TN; t++) {
            STEEL_PRAGMA_UNROLL
            for (short r = 0; r < 2; r++) {
              const uint q = ws[t][r][f / 2] >> (16 * (f % 2));
              const half sc = (f < 4) ? sc0[t][r] : sc1[t][r];
              const vec<Wtype, 4> v = dequantize_fp4_x4_scaled<Wtype>(q, sc);
              STEEL_PRAGMA_UNROLL
              for (short j = 0; j < 4; j++) {
                b[t][4 * r + j] = v[j];
              }
            }
          }
          STEEL_PRAGMA_UNROLL
          for (short t = 0; t < TN; t += 2) {
            Frag::mma(
                Dtile.frag_at(0, t),
                Dtile.frag_at(0, t + 1),
                a,
                metal::bool_constant<false>{},
                b[t],
                b[t + 1],
                metal::bool_constant<true>{});
          }
        }
      }
    }
  } else {
  for (int k = simd_gid * BK; k < K; k += WK * BK) {
    vec<T, 8> xs[2][4];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; r++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < 4; i++) {
        xs[r][i] = x_rows[r]
            ? *(const device vec<T, 8>*)(x + r * 8 * static_cast<int64_t>(K) +
                                         k + 8 * i)
            : vec<T, 8>(0);
      }
    }
    uint4 ws[TN][2][loads];
    scale_bits_t ss[TN][2];
    STEEL_PRAGMA_UNROLL
    for (short t = 0; t < TN; t++) {
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 2; r++) {
        const int row = 16 * t + 8 * r;
        const device uint8_t* wt =
            wl + row * static_cast<int64_t>(row_bytes) + k / pack_factor;
        const device uint8_t* st =
            scales + row * static_cast<int64_t>(row_scales) + k / group_size;
        STEEL_PRAGMA_UNROLL
        for (short l = 0; l < loads; l++) {
          ws[t][r][l] = *(const device uint4*)(wt + 16 * l);
        }
        ss[t][r] = *(const device scale_bits_t*)st;
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < TK; f++) {
      typename Frag::template dtype_frag_t<T> a;
      STEEL_PRAGMA_UNROLL
      for (short r = 0; r < 2; r++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j++) {
          a[4 * r + j] = xs[r][f / 2][4 * (f % 2) + j];
        }
      }
      typename Frag::template dtype_frag_t<Wtype> b[TN];
      STEEL_PRAGMA_UNROLL
      for (short t = 0; t < TN; t++) {
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; r++) {
          uint q;
          if constexpr (bits == 4) {
            q = ws[t][r][0][f / 2] >> (16 * (f % 2));
          } else {
            q = ws[t][r][f / 4][f % 4];
          }
          uint8_t s = ss[t][r];
          if constexpr (group_size == 16) {
            s = ss[t][r] >> (8 * (f / 4));
          }
          const vec<Wtype, 4> v =
              dequantize_fp_x4<Wtype, group_size, bits>(q, s);
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < 4; j++) {
            b[t][4 * r + j] = v[j];
          }
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short t = 0; t < TN; t += 2) {
        Frag::mma(
            Dtile.frag_at(0, t),
            Dtile.frag_at(0, t + 1),
            a,
            metal::bool_constant<false>{},
            b[t],
            b[t + 1],
            metal::bool_constant<true>{});
      }
    }
  }
  }

  const ushort simd_lid = __metal_get_thread_index_in_simdgroup(ushort());
  threadgroup float4 Ps[(WK / 2) * SIMD_SIZE * TN];
  for (short half_pass = 0; half_pass < 2; ++half_pass) {
    if (half_pass == 1 && M <= 8) {
      break;
    }
    const short base_e = half_pass * 4;
    STEEL_PRAGMA_UNROLL
    for (int stride = WK / 2; stride > 0; stride >>= 1) {
      if (simd_gid >= uint(stride) && simd_gid < uint(stride * 2)) {
        threadgroup float4* dst =
            Ps + ((simd_gid - stride) * SIMD_SIZE + simd_lid) * TN;
        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < TN; ++t) {
          const auto f = Dtile.frag_at(0, t);
          dst[t] = float4(
              f[base_e + 0], f[base_e + 1], f[base_e + 2], f[base_e + 3]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid < uint(stride)) {
        const threadgroup float4* src =
            Ps + (simd_gid * SIMD_SIZE + simd_lid) * TN;
        STEEL_PRAGMA_UNROLL
        for (short t = 0; t < TN; ++t) {
          auto& f = Dtile.frag_at(0, t);
          const float4 p = src[t];
          f[base_e + 0] += p.x;
          f[base_e + 1] += p.y;
          f[base_e + 2] += p.z;
          f[base_e + 3] += p.w;
        }
      }
      if (stride > 1 || (half_pass == 0 && M > 8)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
  }
  if (simd_gid == 0) {
    Dtile.store_rows(y, N, M);
  }
}
