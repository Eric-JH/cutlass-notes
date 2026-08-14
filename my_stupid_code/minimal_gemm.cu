#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

namespace spec{
using namespace cute;

template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8>
struct KernelSpec
{
  using T = T_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = SM80_16x8x8_F16F16F16F16_TN;
  using TiledMMA = decltype(make_tile_mma(MMA_op));

  static constexpr kThreadNum = size(TiledMMA{});
  static constexpr kShmSize = 0;
};
}

template<typename Spec, bool IsGemm>
__global__ void minimal_gemm(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k){
    using namespace cute;

    using X = Underscore;
    using T = typename Spec::T;
    using TiledMMA = typename Spec::TiledMMA;

    constexpr int kTileM = Spec::kTileM;
    constexpr int kTileN = Spec::kTileN;
    constexpr int kTileK = Spec::kTileK;

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, _1));
    Tensor mB = make_tensor(make_gmem_ptr((T *)Bprt), make_shape(n, k), make_stride(k, _1));
    Tensor mC = make_tensor(make_gmem_ptr((T *)Cptr), make_shape(m, n), make_stride(n, _1));

    auto Tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(tid);

    Tensor tCgA = thr_mma.partition_A(gA);
    Tensor tCgB = thr_mma.partition_B(gB);
    Tensor tCgC = thr_mma.partition_C(gC);

    Tensor tCrA = thr_mma.partition_fragment_A(gA);
    Tensor tCrB = thr_mma.partition_fragment_B(gB);
    Tensor tCrC = thr_mma.partition_fragment_C(gC);

    auto copy_atom = AutoVectorizingCopy{};

    copy(copy_atom, tCgA, tCrA);
    copy(copy_atom, tCgB, tCrB);

    if constexpr (IsGemm)
      clear(tCrC);
    else
      copy(copy_atom, tCgC, tCrC);

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    copy(copy_atom, tCrC, tCgC);
}

#define CHECK_TORCH_TENSOR_DTYPE(T, DTYPE)                                                                            
  do {                                                                                                                
    if ((T).options().dtype() != (DTYPE)) {                                                                           
      std::cerr << "Tensor dtype mismatch! Expected: " << (DTYPE) << ", but got: " << (T).options().dtype() << " at"  
                << __FILE__ << ": " << __LINE__ << std::endl;                                                         
      std::exit();
    }
  } while (0);

