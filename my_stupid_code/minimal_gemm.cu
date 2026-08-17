#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>
#include "helper.h"

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
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}));

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kShmSize = 0;
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

    Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr((T *)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
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

template <typename ComputeType, typename AccType = ComputeType>
torch::Tensor run_minimal_gemm(const torch::Tensor &a, const torch::Tensor &b, std::optional<torch::Tensor> &_c) {
  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  const int M = 16;
  const int N = 8;
  const int K = 8;

  auto torch_compute_type = [] {
    if constexpr (std::is_same_v<ComputeType, cute::half_t>) return torch::kHalf;
    throw std::runtime_error("Unsupported ComputeType!");
  }();

  auto torch_acc_type = [] {
    if constexpr (std::is_same_v<AccType, cute::half_t>) return torch::kHalf;
    throw std::runtime_error("Unsupported AccType!");
  }();

  torch::Tensor c;
  bool is_gemm;

  if (!_c.has_value()) {
    auto options = torch::TensorOptions().dtype(torch_acc_type).device(torch::kCUDA);
    c = torch::empty({M, N}, options);
    is_gemm = true;
  }
  else {
    c = _c.value();
    is_gemm = false;
  }
  
  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_acc_type)
  
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  using Spec = spec::KernelSpec<ComputeType, M, N, K>;
  
  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
        block.z, grid.x, grid.y, grid.z, shm_size);
  
  cudaDeviceSynchronize();

  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    minimal_gemm<Spec, IsGemm><<<grid, block, shm_size, stream>>>(
      reinterpret_cast<AccType *>(c.data_ptr()), reinterpret_cast<ComputeType *>(a.data_ptr()),
      reinterpret_cast<ComputeType *>(b.data_ptr()), M, N, K
    );
  });

  cudaDeviceSynchronize();
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minimal_gemm", &(run_minimal_gemm<cute::half_t>), "Run a single 16x8x8 MMA operation.");
}
