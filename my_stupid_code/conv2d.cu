#include "helper.h"

// Implicit-GEMM Conv2D
// -------------------
//   x : (H, W, C)        half   -- NHWC activation, contiguous
//   w : (N, KH, KW, C)   half   -- OIHW weight, contiguous (K = KH*KW*C)
//   y : (H, W, N)        half   -- NHWC output, contiguous
//
// Every CTA owns a (kBlockM x kBlockN) tile of the implicit GEMM
//   y[m, n] = sum_k im2col(x)[m, k] * w[n, k],  k = ((kh*KW)+kw)*C + c.
// The im2col tile is materialized in shared memory, then fed to the tensor
// cores through the same TiledCopy -> TiledMMA dataflow used by the GEMM
// examples.  K is zero-padded up to a multiple of the MMA K tile.

namespace spec {
using namespace cute;

template <typename ComputeTypeA_,
          typename ComputeTypeC_,
          typename OutType_,
          int H_,
          int W_,
          int C_,
          int N_,
          int KH_,
          int KW_,
          int PadH_,
          int PadW_,
          int kBlockM_ = 128>
struct Conv2dSpec {
  using ComputeTypeA = ComputeTypeA_; // activation / weight dtype
  using ComputeTypeB = ComputeTypeA_;
  using ComputeTypeC = ComputeTypeC_; // accumulator dtype
  using OutType = OutType_;

  static constexpr int H = H_;
  static constexpr int W = W_;
  static constexpr int C = C_;
  static constexpr int N = N_;
  static constexpr int KH = KH_;
  static constexpr int KW = KW_;
  static constexpr int PadH = PadH_;
  static constexpr int PadW = PadW_;

  static constexpr int M = H * W;
  static constexpr int KReal = KH * KW * C;

  using MMA_op = std::conditional_t<std::is_same_v<ComputeTypeC, float>,
                                    SM80_16x8x8_F32F16F16F32_TN,
                                    SM80_16x8x8_F16F16F16F16_TN>;
  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_shape = typename MMA_traits::Shape_MNK; // (16, 8, 8)

  // 4 warps laid out over (M, N).  N == 8 keeps all four warps on M, otherwise
  // two warps cover M and two cover N; the block tile stays 128 x N.
  static constexpr int kMmaThrExpandN = (N >= 16) ? 2 : 1;
  static constexpr int kMmaThrExpandM = 4 / kMmaThrExpandN;
  static constexpr int kMmaThrExpandK = 1;

  // Value expansion fills out the 128 x N block tile.
  static constexpr int kMmaValExpandM = kBlockM_ / (kMmaThrExpandM * get<0>(MMA_shape{}));
  static constexpr int kMmaValExpandN = N / (kMmaThrExpandN * get<1>(MMA_shape{}));
  static constexpr int kMmaValExpandK = 1;

  static_assert(N == 8 || N % 16 == 0, "N must be 8 or a multiple of 16");
  static_assert(kBlockM_ % (kMmaThrExpandM * get<0>(MMA_shape{})) == 0, "block M must divide the warp layout");

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{}); // 128
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{}); // N
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{}); // 8

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = N;
  static constexpr int kBlockK = (KReal + kMmaTileK - 1) / kMmaTileK * kMmaTileK; // padded K

  static_assert(kBlockM == kMmaTileM, "block M must match the tiled MMA M extent");
  static_assert(kBlockN == kMmaTileN, "block N must match the tiled MMA N extent");
  static_assert(kBlockK % kMmaTileK == 0, "block K must be a multiple of the MMA K tile");
  static_assert(M % kBlockM == 0, "M must be divisible by the block M tile");

  using MMAThrLayout =
      decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

  using Copy_S2R_op = AutoVectorizingCopy;
  using CopyA_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeA>;
  using CopyB_S2R_atom = Copy_Atom<Copy_S2R_op, ComputeTypeB>;
  using TiledCopyA_S2R = decltype(make_tiled_copy_A(CopyA_S2R_atom{}, TiledMMA{}));
  using TiledCopyB_S2R = decltype(make_tiled_copy_B(CopyB_S2R_atom{}, TiledMMA{}));

  using SmemLayoutA = decltype(
      make_layout(make_shape(Int<kBlockM>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));
  using SmemLayoutB = decltype(
      make_layout(make_shape(Int<kBlockN>{}, Int<kBlockK>{}), make_stride(Int<kBlockK>{}, Int<1>{})));

  static constexpr int kShmSizeA = cosize(SmemLayoutA{}) * sizeof(ComputeTypeA);
  static constexpr int kShmSizeB = cosize(SmemLayoutB{}) * sizeof(ComputeTypeB);
  static constexpr int kShmSize = kShmSizeA + kShmSizeB;

  static constexpr int kThreadNum = size(TiledMMA{});
};
} // namespace spec

// Gather one im2col element: k = (kh*KW + kw)*C + c, zero outside the padded image.
template <typename T, int H, int W, int C, int KH, int KW, int PadH, int PadW, int KReal>
CUTE_HOST_DEVICE T im2col_at(const T *x, int m, int k) {
  if (k >= KReal) {
    return T(0);
  }
  int kh = k / (KW * C);
  int rem = k - kh * (KW * C);
  int kw = rem / C;
  int c = rem - kw * C;
  int h = m / W;
  int w = m - h * W;
  int ih = h + kh - PadH;
  int iw = w + kw - PadW;
  if (ih < 0 || ih >= H || iw < 0 || iw >= W) {
    return T(0);
  }
  return x[(ih * W + iw) * C + c];
}

template <typename Spec>
__global__ void conv2d_implicit_gemm(const void *__restrict__ Xptr,
                                     const void *__restrict__ Wptr,
                                     void *__restrict__ Yptr) {
  using namespace cute;

  using X = Underscore;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using OutType = typename Spec::OutType;
  using TiledMMA = typename Spec::TiledMMA;
  using TiledCopyA_S2R = typename Spec::TiledCopyA_S2R;
  using TiledCopyB_S2R = typename Spec::TiledCopyB_S2R;
  using SmemLayoutA = typename Spec::SmemLayoutA;
  using SmemLayoutB = typename Spec::SmemLayoutB;

  constexpr int H = Spec::H;
  constexpr int W = Spec::W;
  constexpr int C = Spec::C;
  constexpr int N = Spec::N;
  constexpr int KH = Spec::KH;
  constexpr int KW = Spec::KW;
  constexpr int PadH = Spec::PadH;
  constexpr int PadW = Spec::PadW;
  constexpr int M = Spec::M;
  constexpr int KReal = Spec::KReal;
  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kThreadNum = Spec::kThreadNum;

  int tid = threadIdx.x;
  int m0 = blockIdx.x * kBlockM;

  extern __shared__ __align__(16) uint8_t smem[];
  ComputeTypeA *sA_ptr = reinterpret_cast<ComputeTypeA *>(smem);
  ComputeTypeB *sB_ptr = reinterpret_cast<ComputeTypeB *>(smem + Spec::kShmSizeA);

  Tensor sA = make_tensor(make_smem_ptr(sA_ptr), SmemLayoutA{}); // (kBlockM, kBlockK)
  Tensor sB = make_tensor(make_smem_ptr(sB_ptr), SmemLayoutB{}); // (kBlockN, kBlockK)

  const ComputeTypeA *x = reinterpret_cast<const ComputeTypeA *>(Xptr);
  const ComputeTypeB *w = reinterpret_cast<const ComputeTypeB *>(Wptr);

  // ---- im2col: one thread builds one pixel row of sA ----
  {
    int m = m0 + tid;
    int h = m / W;
    int ww = m - h * W;

#pragma unroll
    for (int kh = 0; kh < KH; ++kh) {
#pragma unroll
      for (int kw = 0; kw < KW; ++kw) {
        int ih = h + kh - PadH;
        int iw = ww + kw - PadW;
        int kbase = (kh * KW + kw) * C;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
          const ComputeTypeA *src = x + (ih * W + iw) * C;
#pragma unroll
          for (int c = 0; c < C; ++c) {
            sA(tid, kbase + c) = src[c];
          }
        } else {
#pragma unroll
          for (int c = 0; c < C; ++c) {
            sA(tid, kbase + c) = ComputeTypeA(0);
          }
        }
      }
    }
#pragma unroll
    for (int k = KReal; k < kBlockK; ++k) {
      sA(tid, k) = ComputeTypeA(0);
    }
  }

  // ---- stage the (tiny) weight tile into shared memory, zero-padding K ----
  {
    constexpr int kNumBElems = kBlockN * kBlockK;
#pragma unroll
    for (int i = tid; i < kNumBElems; i += kThreadNum) {
      int n = i / kBlockK;
      int k = i - n * kBlockK;
      sB(n, k) = (k < KReal) ? w[n * KReal + k] : ComputeTypeB(0);
    }
  }

  __syncthreads();

  Tensor mY = make_tensor(make_gmem_ptr(reinterpret_cast<OutType *>(Yptr)), make_shape(M, Int<N>{}),
                          make_stride(Int<N>{}, Int<1>{})); // (M, N)

  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(blockIdx.x, 0, 0);
  Tensor gY = local_tile(mY, tiler, coord, Step<_1, _1, X>{}); // (kBlockM, kBlockN)

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgY = thr_mma.partition_C(gY);          // (MMA, MMA_M, MMA_N)
  Tensor tCrC = thr_mma.partition_fragment_C(gY); // (MMA, MMA_M, MMA_N)
  Tensor tCrA = thr_mma.partition_fragment_A(sA); // (MMA, MMA_M, MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB); // (MMA, MMA_N, MMA_K)

  TiledCopyA_S2R s2r_tiled_copy_a;
  ThrCopy s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
  Tensor tAsA = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K)
  Tensor tArA = s2r_thr_copy_a.retile_D(tCrA);  // (CPY, CPY_M, CPY_K)

  TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_N, CPY_K)
  Tensor tBrB = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_N, CPY_K)

  copy(s2r_tiled_copy_a, tAsA, tArA);
  copy(s2r_tiled_copy_b, tBsB, tBrB);

  clear(tCrC);
  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  Tensor tCrO = make_tensor_like<OutType>(tCrC);
  copy(tCrC, tCrO); // precision conversion

  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCrO, tCgY);
}
// Implicit GEMM conv: no im2col tile in shared memory.  The A operand is
// gathered straight from global memory into the MMA register fragment; only
// the small weight tile is staged in shared memory.
template <typename Spec>
__global__ void conv2d_implicit(const void *__restrict__ Xptr,
                                const void *__restrict__ Wptr,
                                void *__restrict__ Yptr) {
  using namespace cute;

  using X = Underscore;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using OutType = typename Spec::OutType;
  using TiledMMA = typename Spec::TiledMMA;
  using TiledCopyB_S2R = typename Spec::TiledCopyB_S2R;
  using SmemLayoutB = typename Spec::SmemLayoutB;

  constexpr int H = Spec::H;
  constexpr int W = Spec::W;
  constexpr int C = Spec::C;
  constexpr int N = Spec::N;
  constexpr int KH = Spec::KH;
  constexpr int KW = Spec::KW;
  constexpr int PadH = Spec::PadH;
  constexpr int PadW = Spec::PadW;
  constexpr int M = Spec::M;
  constexpr int KReal = Spec::KReal;
  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kThreadNum = Spec::kThreadNum;

  int tid = threadIdx.x;
  int m0 = blockIdx.x * kBlockM;

  extern __shared__ __align__(16) uint8_t smem[];
  ComputeTypeB *sB_ptr = reinterpret_cast<ComputeTypeB *>(smem);
  Tensor sB = make_tensor(make_smem_ptr(sB_ptr), SmemLayoutB{}); // (kBlockN, kBlockK)

  const ComputeTypeA *x = reinterpret_cast<const ComputeTypeA *>(Xptr);
  const ComputeTypeB *w = reinterpret_cast<const ComputeTypeB *>(Wptr);

  {
    constexpr int kNumBElems = kBlockN * kBlockK;
#pragma unroll
    for (int i = tid; i < kNumBElems; i += kThreadNum) {
      int n = i / kBlockK;
      int k = i - n * kBlockK;
      sB(n, k) = (k < KReal) ? w[n * KReal + k] : ComputeTypeB(0);
    }
  }
  __syncthreads();

  Tensor mY = make_tensor(make_gmem_ptr(reinterpret_cast<OutType *>(Yptr)), make_shape(M, Int<N>{}),
                          make_stride(Int<N>{}, Int<1>{})); // (M, N)
  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(blockIdx.x, 0, 0);
  Tensor gY = local_tile(mY, tiler, coord, Step<_1, _1, X>{}); // (kBlockM, kBlockN)

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgY = thr_mma.partition_C(gY);
  Tensor tCrC = thr_mma.partition_fragment_C(gY);

  // A fragment layout is derived from a shape-only tensor (never read), then
  // filled element-by-element with the im2col gather.
  Tensor mA_dummy = make_tensor(make_gmem_ptr(reinterpret_cast<ComputeTypeA *>(Yptr)),
                                make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                                make_stride(Int<kBlockK>{}, Int<1>{}));
  Tensor tCrA = thr_mma.partition_fragment_A(mA_dummy); // (MMA, MMA_M, MMA_K)

  Tensor cA = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockK>{}));
  Tensor tCcA = thr_mma.partition_A(cA); // (MMA, MMA_M, MMA_K) -> (m, k)

#pragma unroll
  for (int km = 0; km < size<2>(tCrA); ++km) {
#pragma unroll
    for (int mm = 0; mm < size<1>(tCrA); ++mm) {
#pragma unroll
      for (int i = 0; i < size<0>(tCrA); ++i) {
        auto cc = tCcA(i, mm, km);
        int mi = int(get<0>(cc));
        int kk = int(get<1>(cc));
        tCrA(i, mm, km) = im2col_at<ComputeTypeA, H, W, C, KH, KW, PadH, PadW, KReal>(x, m0 + mi, kk);
      }
    }
  }

  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tBrB = s2r_thr_copy_b.retile_D(tCrB);
  copy(s2r_tiled_copy_b, tBsB, tBrB);

  clear(tCrC);
  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  Tensor tCrO = make_tensor_like<OutType>(tCrC);
  copy(tCrC, tCrO);

  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCrO, tCgY);
}
// Implicit GEMM conv with the raw input patch staged in shared memory.
// kBlockM is chosen to divide W (here 96 | 960), so every CTA covers a
// contiguous segment of a single row and the patch is a clean
// (KH x (kBlockM + KW - 1) x C) rectangle.  The im2col indexing happens
// on the smem->register path instead of being materialized in smem.
template <typename Spec>
__global__ void conv2d_implicit_smem(const void *__restrict__ Xptr,
                                     const void *__restrict__ Wptr,
                                     void *__restrict__ Yptr) {
  using namespace cute;

  using X = Underscore;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using OutType = typename Spec::OutType;
  using TiledMMA = typename Spec::TiledMMA;
  using TiledCopyB_S2R = typename Spec::TiledCopyB_S2R;
  using SmemLayoutB = typename Spec::SmemLayoutB;

  constexpr int H = Spec::H;
  constexpr int W = Spec::W;
  constexpr int C = Spec::C;
  constexpr int N = Spec::N;
  constexpr int KH = Spec::KH;
  constexpr int KW = Spec::KW;
  constexpr int PadH = Spec::PadH;
  constexpr int PadW = Spec::PadW;
  constexpr int M = Spec::M;
  constexpr int KReal = Spec::KReal;
  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kThreadNum = Spec::kThreadNum;

  static_assert(W % kBlockM == 0, "patch version requires kBlockM to divide W");

  constexpr int kPatchW = kBlockM + KW - 1;
  constexpr int kPatchH = KH;
  constexpr int kPatchPixels = kPatchH * kPatchW;

  int tid = threadIdx.x;
  int m0 = blockIdx.x * kBlockM;
  int h0 = m0 / W;
  int w0 = m0 - h0 * W;

  extern __shared__ __align__(16) uint8_t smem[];
  ComputeTypeA *sPatch_ptr = reinterpret_cast<ComputeTypeA *>(smem);
  ComputeTypeB *sB_ptr = reinterpret_cast<ComputeTypeB *>(smem + kPatchPixels * C * sizeof(ComputeTypeA));

  Tensor sPatch = make_tensor(
      make_smem_ptr(sPatch_ptr),
      make_layout(make_shape(Int<kPatchH>{}, Int<kPatchW>{}, Int<C>{}),
                  make_stride(Int<kPatchW * C>{}, Int<C>{}, Int<1>{}))); // (PH, PW, C)
  Tensor sB = make_tensor(make_smem_ptr(sB_ptr), SmemLayoutB{});          // (kBlockN, kBlockK)

  const ComputeTypeA *x = reinterpret_cast<const ComputeTypeA *>(Xptr);
  const ComputeTypeB *w = reinterpret_cast<const ComputeTypeB *>(Wptr);

  // Stage the raw patch: one pixel = C contiguous halfs, coalesced.
#pragma unroll
  for (int p = tid; p < kPatchPixels; p += kThreadNum) {
    int col = p % kPatchW;
    int row = p / kPatchW;
    int ih = h0 + row - PadH;
    int iw = w0 + col - PadW;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
      const ComputeTypeA *src = x + (ih * W + iw) * C;
#pragma unroll
      for (int c = 0; c < C; ++c) {
        sPatch(row, col, c) = src[c];
      }
    } else {
#pragma unroll
      for (int c = 0; c < C; ++c) {
        sPatch(row, col, c) = ComputeTypeA(0);
      }
    }
  }

  {
    constexpr int kNumBElems = kBlockN * kBlockK;
#pragma unroll
    for (int i = tid; i < kNumBElems; i += kThreadNum) {
      int n = i / kBlockK;
      int k = i - n * kBlockK;
      sB(n, k) = (k < KReal) ? w[n * KReal + k] : ComputeTypeB(0);
    }
  }
  __syncthreads();

  Tensor mY = make_tensor(make_gmem_ptr(reinterpret_cast<OutType *>(Yptr)), make_shape(M, Int<N>{}),
                          make_stride(Int<N>{}, Int<1>{})); // (M, N)
  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(blockIdx.x, 0, 0);
  Tensor gY = local_tile(mY, tiler, coord, Step<_1, _1, X>{}); // (kBlockM, kBlockN)

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgY = thr_mma.partition_C(gY);
  Tensor tCrC = thr_mma.partition_fragment_C(gY);

  Tensor mA_dummy = make_tensor(make_gmem_ptr(reinterpret_cast<ComputeTypeA *>(Yptr)),
                                make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                                make_stride(Int<kBlockK>{}, Int<1>{}));
  Tensor tCrA = thr_mma.partition_fragment_A(mA_dummy); // (MMA, MMA_M, MMA_K)

  Tensor cA = make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockK>{}));
  Tensor tCcA = thr_mma.partition_A(cA); // (MMA, MMA_M, MMA_K) -> (m, k)

  // Build the A fragment by gathering from the smem patch.
#pragma unroll
  for (int km = 0; km < size<2>(tCrA); ++km) {
#pragma unroll
    for (int mm = 0; mm < size<1>(tCrA); ++mm) {
#pragma unroll
      for (int i = 0; i < size<0>(tCrA); ++i) {
        auto cc = tCcA(i, mm, km);
        int ml = int(get<0>(cc));
        int kk = int(get<1>(cc));
        ComputeTypeA v = ComputeTypeA(0);
        if (kk < KReal) {
          int kh = kk / (KW * C);
          int rem = kk - kh * (KW * C);
          int kw = rem / C;
          int c = rem - kw * C;
          v = sPatch(kh, ml + kw, c);
        }
        tCrA(i, mm, km) = v;
      }
    }
  }

  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  TiledCopyB_S2R s2r_tiled_copy_b;
  ThrCopy s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
  Tensor tBsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tBrB = s2r_thr_copy_b.retile_D(tCrB);
  copy(s2r_tiled_copy_b, tBsB, tBrB);

  clear(tCrC);
  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  Tensor tCrO = make_tensor_like<OutType>(tCrC);
  copy(tCrC, tCrO);

  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCrO, tCgY);
}
// cp.async double-buffered pipeline for the materialized-im2col path.
// K = 40 is split into kTilesK tiles of kMmaTileK (=8, i.e. 2 taps each).
// sA is double buffered, so the gather of tile kt+1 is issued with cp.async
// while the tensor cores consume tile kt.
template <typename Spec>
__global__ void conv2d_pipeline(const void *__restrict__ Xptr,
                                const void *__restrict__ Wptr,
                                void *__restrict__ Yptr) {
  using namespace cute;

  using X = Underscore;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using OutType = typename Spec::OutType;
  using TiledMMA = typename Spec::TiledMMA;
  using TiledCopyA_S2R = typename Spec::TiledCopyA_S2R;
  using TiledCopyB_S2R = typename Spec::TiledCopyB_S2R;
  using SmemLayoutB = typename Spec::SmemLayoutB;

  constexpr int H = Spec::H;
  constexpr int W = Spec::W;
  constexpr int C = Spec::C;
  constexpr int N = Spec::N;
  constexpr int KH = Spec::KH;
  constexpr int KW = Spec::KW;
  constexpr int PadH = Spec::PadH;
  constexpr int PadW = Spec::PadW;
  constexpr int M = Spec::M;
  constexpr int KReal = Spec::KReal;
  constexpr int kBlockM = Spec::kBlockM;
  constexpr int kBlockN = Spec::kBlockN;
  constexpr int kBlockK = Spec::kBlockK;
  constexpr int kThreadNum = Spec::kThreadNum;
  constexpr int kTileK = Spec::kMmaTileK;
  constexpr int kTilesK = kBlockK / kTileK;
  constexpr int kTapsPerTile = kTileK / C;

  static_assert(kBlockK == kTilesK * kTileK, "K must be a whole number of MMA K tiles");
  static_assert(kTapsPerTile * C == kTileK, "each K tile must cover a whole number of taps");

  int tid = threadIdx.x;
  int m = blockIdx.x * kBlockM + tid;
  int h = m / W;
  int ww = m - h * W;

  extern __shared__ __align__(16) uint8_t smem[];
  ComputeTypeA *sA_ptr = reinterpret_cast<ComputeTypeA *>(smem);
  ComputeTypeB *sB_ptr = reinterpret_cast<ComputeTypeB *>(smem + 2 * kBlockM * kTileK * sizeof(ComputeTypeA));

  Tensor sA = make_tensor(make_smem_ptr(sA_ptr),
                          make_layout(make_shape(Int<2>{}, Int<kBlockM>{}, Int<kTileK>{}),
                                      make_stride(Int<kBlockM * kTileK>{}, Int<kTileK>{}, Int<1>{}))); // (2, M, Ktile)
  Tensor sB = make_tensor(make_smem_ptr(sB_ptr), SmemLayoutB{});

  const ComputeTypeA *x = reinterpret_cast<const ComputeTypeA *>(Xptr);
  const ComputeTypeB *w = reinterpret_cast<const ComputeTypeB *>(Wptr);

  {
    constexpr int kNumBElems = kBlockN * kBlockK;
#pragma unroll
    for (int i = tid; i < kNumBElems; i += kThreadNum) {
      int n = i / kBlockK;
      int k = i - n * kBlockK;
      sB(n, k) = (k < KReal) ? w[n * KReal + k] : ComputeTypeB(0);
    }
  }
  __syncthreads();

  Tensor mY = make_tensor(make_gmem_ptr(reinterpret_cast<OutType *>(Yptr)), make_shape(M, Int<N>{}),
                          make_stride(Int<N>{}, Int<1>{}));
  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(blockIdx.x, 0, 0);
  Tensor gY = local_tile(mY, tiler, coord, Step<_1, _1, X>{});

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgY = thr_mma.partition_C(gY);
  Tensor tCrC = thr_mma.partition_fragment_C(gY);
  clear(tCrC);

  Tensor mA_dummy = make_tensor(make_gmem_ptr(reinterpret_cast<ComputeTypeA *>(Yptr)),
                                make_shape(Int<kBlockM>{}, Int<kBlockK>{}),
                                make_stride(Int<kBlockK>{}, Int<1>{}));
  Tensor tCrA = thr_mma.partition_fragment_A(mA_dummy); // (MMA, MMA_M, MMA_K)

  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  TiledCopyB_S2R s2r_b;
  ThrCopy s2r_thr_b = s2r_b.get_slice(tid);
  Tensor tBsB = s2r_thr_b.partition_S(sB);
  Tensor tBrB = s2r_thr_b.retile_D(tCrB);
  copy(s2r_b, tBsB, tBrB); // whole B fragment loaded once

  // Issue cp.async for one K tile (kTapsPerTile taps x C channels per pixel).
  auto prefetch = [&](int kt, int buf) {
#pragma unroll
    for (int t = 0; t < kTapsPerTile; ++t) {
      int tap = kt * kTapsPerTile + t;
      bool tap_ok = tap < KH * KW;
      int kh = tap_ok ? tap / KW : 0;
      int kw = tap_ok ? tap - (tap / KW) * KW : 0;
      int ih = h + kh - PadH;
      int iw = ww + kw - PadW;
      bool pred = tap_ok && ih >= 0 && ih < H && iw >= 0 && iw < W;
      const uint2 *src = reinterpret_cast<const uint2 *>(x + (pred ? (ih * W + iw) * C : 0));
      uint2 *dst = reinterpret_cast<uint2 *>(&sA(buf, tid, t * C));
      SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint2>::copy(*src, *dst, pred);
    }
    cp_async_fence();
  };

  prefetch(0, 0);

#pragma unroll
  for (int kt = 0; kt < kTilesK; ++kt) {
    if (kt + 1 < kTilesK) {
      prefetch(kt + 1, (kt + 1) & 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    TiledCopyA_S2R s2r_a;
    ThrCopy s2r_thr_a = s2r_a.get_slice(tid);
    Tensor sAb = sA(kt & 1, _, _);            // (kBlockM, kTileK)
    Tensor tAsA = s2r_thr_a.partition_S(sAb); // (CPY, CPY_M, CPY_K)
    Tensor tArA = s2r_thr_a.retile_D(tCrA);   // (CPY, CPY_M, CPY_K)
    copy(s2r_a, tAsA(_, _, 0), tArA(_, _, kt));

    gemm(tiled_mma, tCrC, tCrA(_, _, kt), tCrB(_, _, kt), tCrC);
    __syncthreads();
  }

  Tensor tCrO = make_tensor_like<OutType>(tCrC);
  copy(tCrC, tCrO);
  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCrO, tCgY);
}

template <int H, int W, int C, int N, int KH, int KW, int PadH, int PadW, typename AccType>
torch::Tensor run_conv2d_impl(const torch::Tensor &x, const torch::Tensor &w) {
  at::cuda::CUDAGuard device_guard{x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  using Spec = spec::Conv2dSpec<cute::half_t, AccType, cute::half_t, H, W, C, N, KH, KW, PadH, PadW>;

  auto y = torch::empty({H, W, N}, x.options());

  dim3 block = Spec::kThreadNum;
  dim3 grid(Spec::M / Spec::kBlockM);
  int shm_size = Spec::kShmSize;

  conv2d_implicit_gemm<Spec><<<grid, block, shm_size, stream>>>(x.data_ptr(), w.data_ptr(), y.data_ptr());

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  }

  return y;
}

template <typename AccType> torch::Tensor run_conv2d(const torch::Tensor &x, const torch::Tensor &w) {
  constexpr int H = 540;
  constexpr int W = 960;
  constexpr int C = 4;
  constexpr int KH = 3;
  constexpr int KW = 3;
  constexpr int PadH = 1;
  constexpr int PadW = 1;

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
  TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be float16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous (H, W, C)");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous (N, KH, KW, C)");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({H, W, C}), "x must have shape (H, W, C) = (", H, ", ", W, ", ", C, ")");
  TORCH_CHECK(w.dim() == 4 && w.size(1) == KH && w.size(2) == KW && w.size(3) == C,
              "w must have shape (N, KH, KW, C) = (N, ", KH, ", ", KW, ", ", C, ")");

  switch (w.size(0)) {
  case 8:
    return run_conv2d_impl<H, W, C, 8, KH, KW, PadH, PadW, AccType>(x, w);
  case 16:
    return run_conv2d_impl<H, W, C, 16, KH, KW, PadH, PadW, AccType>(x, w);
  case 32:
    return run_conv2d_impl<H, W, C, 32, KH, KW, PadH, PadW, AccType>(x, w);
  case 64:
    return run_conv2d_impl<H, W, C, 64, KH, KW, PadH, PadW, AccType>(x, w);
  default:
    TORCH_CHECK(false, "N must be one of {8, 16, 32, 64}, got ", w.size(0));
  }
}
template <int H, int W, int C, int N, int KH, int KW, int PadH, int PadW, typename AccType>
torch::Tensor run_conv2d_implicit_impl(const torch::Tensor &x, const torch::Tensor &w) {
  at::cuda::CUDAGuard device_guard{x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  using Spec = spec::Conv2dSpec<cute::half_t, AccType, cute::half_t, H, W, C, N, KH, KW, PadH, PadW>;

  auto y = torch::empty({H, W, N}, x.options());

  dim3 block = Spec::kThreadNum;
  dim3 grid(Spec::M / Spec::kBlockM);
  int shm_size = Spec::kShmSizeB; // only the weight tile is staged

  conv2d_implicit<Spec><<<grid, block, shm_size, stream>>>(x.data_ptr(), w.data_ptr(), y.data_ptr());

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  }

  return y;
}

template <typename AccType> torch::Tensor run_conv2d_implicit(const torch::Tensor &x, const torch::Tensor &w) {
  constexpr int H = 540;
  constexpr int W = 960;
  constexpr int C = 4;
  constexpr int KH = 3;
  constexpr int KW = 3;
  constexpr int PadH = 1;
  constexpr int PadW = 1;

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
  TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be float16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous (H, W, C)");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous (N, KH, KW, C)");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({H, W, C}), "x must have shape (H, W, C)");
  TORCH_CHECK(w.dim() == 4 && w.size(1) == KH && w.size(2) == KW && w.size(3) == C,
              "w must have shape (N, KH, KW, C)");

  switch (w.size(0)) {
  case 8:
    return run_conv2d_implicit_impl<H, W, C, 8, KH, KW, PadH, PadW, AccType>(x, w);
  case 16:
    return run_conv2d_implicit_impl<H, W, C, 16, KH, KW, PadH, PadW, AccType>(x, w);
  case 32:
    return run_conv2d_implicit_impl<H, W, C, 32, KH, KW, PadH, PadW, AccType>(x, w);
  case 64:
    return run_conv2d_implicit_impl<H, W, C, 64, KH, KW, PadH, PadW, AccType>(x, w);
  default:
    TORCH_CHECK(false, "N must be one of {8, 16, 32, 64}, got ", w.size(0));
  }
}
template <int H, int W, int C, int N, int KH, int KW, int PadH, int PadW, typename AccType>
torch::Tensor run_conv2d_implicit_smem_impl(const torch::Tensor &x, const torch::Tensor &w) {
  at::cuda::CUDAGuard device_guard{x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int kBlockM = 192; // divides W = 960, keeps each CTA inside one row
  using Spec = spec::Conv2dSpec<cute::half_t, AccType, cute::half_t, H, W, C, N, KH, KW, PadH, PadW, kBlockM>;

  auto y = torch::empty({H, W, N}, x.options());

  constexpr int kPatchPixels = KH * (kBlockM + KW - 1);
  constexpr int kShmSize = kPatchPixels * C * sizeof(cute::half_t) + Spec::kShmSizeB;

  dim3 block = Spec::kThreadNum;
  dim3 grid(Spec::M / kBlockM);
  int shm_size = kShmSize;

  conv2d_implicit_smem<Spec><<<grid, block, shm_size, stream>>>(x.data_ptr(), w.data_ptr(), y.data_ptr());

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  }

  return y;
}

template <typename AccType> torch::Tensor run_conv2d_implicit_smem(const torch::Tensor &x, const torch::Tensor &w) {
  constexpr int H = 540;
  constexpr int W = 960;
  constexpr int C = 4;
  constexpr int KH = 3;
  constexpr int KW = 3;
  constexpr int PadH = 1;
  constexpr int PadW = 1;

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
  TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be float16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous (H, W, C)");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous (N, KH, KW, C)");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({H, W, C}), "x must have shape (H, W, C)");
  TORCH_CHECK(w.dim() == 4 && w.size(1) == KH && w.size(2) == KW && w.size(3) == C,
              "w must have shape (N, KH, KW, C)");

  switch (w.size(0)) {
  case 8:
    return run_conv2d_implicit_smem_impl<H, W, C, 8, KH, KW, PadH, PadW, AccType>(x, w);
  case 16:
    return run_conv2d_implicit_smem_impl<H, W, C, 16, KH, KW, PadH, PadW, AccType>(x, w);
  case 32:
    return run_conv2d_implicit_smem_impl<H, W, C, 32, KH, KW, PadH, PadW, AccType>(x, w);
  case 64:
    return run_conv2d_implicit_smem_impl<H, W, C, 64, KH, KW, PadH, PadW, AccType>(x, w);
  default:
    TORCH_CHECK(false, "N must be one of {8, 16, 32, 64}, got ", w.size(0));
  }
}
template <int H, int W, int C, int N, int KH, int KW, int PadH, int PadW, typename AccType>
torch::Tensor run_conv2d_pipeline_impl(const torch::Tensor &x, const torch::Tensor &w) {
  at::cuda::CUDAGuard device_guard{x.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  using Spec = spec::Conv2dSpec<cute::half_t, AccType, cute::half_t, H, W, C, N, KH, KW, PadH, PadW>;

  auto y = torch::empty({H, W, N}, x.options());

  constexpr int kDoubleBufferBytes = 2 * Spec::kBlockM * Spec::kMmaTileK * sizeof(cute::half_t);
  constexpr int kShmSize = kDoubleBufferBytes + Spec::kShmSizeB;

  dim3 block = Spec::kThreadNum;
  dim3 grid(Spec::M / Spec::kBlockM);
  int shm_size = kShmSize;

  conv2d_pipeline<Spec><<<grid, block, shm_size, stream>>>(x.data_ptr(), w.data_ptr(), y.data_ptr());

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
  }

  return y;
}

template <typename AccType> torch::Tensor run_conv2d_pipeline(const torch::Tensor &x, const torch::Tensor &w) {
  constexpr int H = 540;
  constexpr int W = 960;
  constexpr int C = 4;
  constexpr int KH = 3;
  constexpr int KW = 3;
  constexpr int PadH = 1;
  constexpr int PadW = 1;

  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be float16");
  TORCH_CHECK(w.scalar_type() == torch::kHalf, "w must be float16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous (H, W, C)");
  TORCH_CHECK(w.is_contiguous(), "w must be contiguous (N, KH, KW, C)");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({H, W, C}), "x must have shape (H, W, C)");
  TORCH_CHECK(w.dim() == 4 && w.size(1) == KH && w.size(2) == KW && w.size(3) == C,
              "w must have shape (N, KH, KW, C)");

  switch (w.size(0)) {
  case 8:
    return run_conv2d_pipeline_impl<H, W, C, 8, KH, KW, PadH, PadW, AccType>(x, w);
  case 16:
    return run_conv2d_pipeline_impl<H, W, C, 16, KH, KW, PadH, PadW, AccType>(x, w);
  case 32:
    return run_conv2d_pipeline_impl<H, W, C, 32, KH, KW, PadH, PadW, AccType>(x, w);
  case 64:
    return run_conv2d_pipeline_impl<H, W, C, 64, KH, KW, PadH, PadW, AccType>(x, w);
  default:
    TORCH_CHECK(false, "N must be one of {8, 16, 32, 64}, got ", w.size(0));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d", &run_conv2d<float>, "Implicit-GEMM 3x3 conv2d, fp16 in/out with fp32 accumulation.");
  m.def("conv2d_f16acc", &run_conv2d<cute::half_t>, "Implicit-GEMM 3x3 conv2d, fp16 in/out with fp16 accumulation.");
  m.def("conv2d_implicit", &run_conv2d_implicit<float>,
        "Implicit-GEMM conv2d with direct gmem->register gather (no im2col smem), fp32 accumulation.");
  m.def("conv2d_implicit_f16acc", &run_conv2d_implicit<cute::half_t>,
        "Implicit-GEMM conv2d with direct gmem->register gather (no im2col smem), fp16 accumulation.");
  m.def("conv2d_implicit_smem", &run_conv2d_implicit_smem<float>,
        "Implicit-GEMM conv2d with raw input patch staged in smem, fp32 accumulation.");
  m.def("conv2d_implicit_smem_f16acc", &run_conv2d_implicit_smem<cute::half_t>,
        "Implicit-GEMM conv2d with raw input patch staged in smem, fp16 accumulation.");
  m.def("conv2d_pipeline", &run_conv2d_pipeline<float>,
        "Materialized im2col with cp.async double-buffered K-tile pipeline, fp32 accumulation.");
  m.def("conv2d_pipeline_f16acc", &run_conv2d_pipeline<cute::half_t>,
        "Materialized im2col with cp.async double-buffered K-tile pipeline, fp16 accumulation.");
}
