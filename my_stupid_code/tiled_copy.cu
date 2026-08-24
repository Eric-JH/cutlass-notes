#include "helper.h"

namespace spec {
using namespace cute;

template <typename OutType_, typename ComputeTypeA_, typename ComputeTypeB_, typename ComputeTypeC_,
            int kTileM_ = 32, int kTileN_ = 32, int kTileK_ = 16>
struct KernelSpec{
  using OutType = OutType_;
  using ComputeTypeA = ComputeTypeA_;
  using ComputeTypeB = ComputeTypeB_;
  using ComputeTypeC = ComputeTypeC_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
  using MMA_traits = MMA_Traits<MMA_op>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = MMA_traits::Shape_MNK;

  static constexpr int kMmaThrExpandM = 2;
  static constexpr int kMmaThrExpandN = 4;
  static constexpr int kMmaThrExpandK = 1;

  static constexpr int kMmaValExpandM = 1;
  static constexpr int kMmaValExpandN = 1;
  static constexpr int kMmaValExpandK = 2;

  static constexpr int kMmaTileM = kMmaThrExpandM * kMmaValExpandM * get<0>(MMA_shape{});
  static constexpr int kMmaTileN = kMmaThrExpandN * kMmaValExpandN * get<1>(MMA_shape{});
  static constexpr int kMmaTileK = kMmaThrExpandK * kMmaValExpandK * get<2>(MMA_shape{});
  using MMAThrLayout = decltype(make_layout(make_shape(Int<kMmaThrExpandM>{}, Int<kMmaThrExpandN>{}, Int<kMmaThrExpandK>{})));
  using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

  using Copy_op = AutoVectorizingCopy;
  using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;
  using CopyB_atom = Copy_Atom<Copy_op, ComputeTypeB>;
  using CopyC_atom = Copy_Atom<Copy_op, ComputeTypeC>;
  using CopyO_atom = Copy_Atom<Copy_op, OutType>;

  using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
  using TiledCopyB = decltype(make_tiled_copy_B(CopyB_atom{}, TiledMMA{}));
  using TiledCopyC = decltype(make_tiled_copy_C(CopyC_atom{}, TiledMMA{}));
  using TiledCopyO = decltype(make_tiled_copy_C(CopyO_atom{}, TiledMMA{}));

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kShmSize = 0;
};
} // namespace spec

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void tiled_copy(void *Cptr, void *Aptr, void *Bptr, int m, int n, int k, void *Outptr) {
  using namespace cute;
  using X = Underscore;
  using ComputeTypeA = typename Spec::ComputeTypeA;
  using ComputeTypeB = typename Spec::ComputeTypeB;
  using ComputeTypeC = typename Spec::ComputeTypeC;
  using OutType = typename Spec::OutType;

  using TiledMMA = typename Spec::TiledMMA;
  using TiledCopyA = typename Spec::TiledCopyA;
  using TiledCopyB = typename Spec::TiledCopyB;
  using TiledCopyC = typename Spec::TiledCopyC;
  using TiledCopyO = typename Spec::TiledCopyO;

  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;

  int tid = threadIdx.x;

  Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
  Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(m, n), make_stride(n, Int<1>{}));

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(0, 0, 0);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});
  Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{});

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);

  Tensor tCgA = thr_mma.partition_A(gA);
  Tensor tCgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.partition_fragment_A(gA);
  Tensor tCrB = thr_mma.partition_fragment_B(gB);
  Tensor tCrC = thr_mma.partition_fragment_C(gC);

  TiledCopyA g2r_tiled_copy_a;
  ThrCopy g2r_thr_copy_a = g2r_tiled_copy_a.get_slice(tid);
  Tensor tAgA = g2r_thr_copy_a.retile_S(tCgA);
  Tensor tArA = g2r_thr_copy_a.retile_D(tCrA);

  TiledCopyB g2r_tiled_copy_b;
  ThrCopy g2r_thr_copy_b = g2r_tiled_copy_b.get_slice(tid);
  Tensor tBgB = g2r_thr_copy_b.retile_S(tCgB);
  Tensor tBrB = g2r_thr_copy_b.retile_D(tCrB);

  copy(g2r_tiled_copy_a, tAgA, tArA);
  copy(g2r_tiled_copy_b, tBgB, tBrB);

  if constexpr (IsGemm) {
    clear(tCrC);
  }
  else {
    TiledCopyC g2r_tiled_copy_c;
    ThrCopy g2r_thr_copy_c = g2r_tiled_copy_c.get_slice(tid);
    Tensor tCgC_g2r = g2r_thr_copy_c.retile_S(tCgC);
    Tensor tCrC_g2r = g2r_thr_copy_c.retile_D(tCrC);
    copy(g2r_tiled_copy_c, tCgC_g2r, tCrC_g2r);
  }

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  TiledCopyO r2g_tiled_copy_o;
  if constexpr (!IsCvtPrecision){
    ThrCopy r2g_thr_copy_o = r2g_tiled_copy_o.get_slice(tid);
    Tensor tCrC_r2g = r2g_thr_copy_o.retile_S(tCrC);
    Tensor tCgC_r2g = r2g_thr_copy_o.retile_D(tCgC);
    copy(r2g_tiled_copy_o, tCrC_r2g, tCgC_r2g);
  }
  else {
    Tensor tCgO = thr_mma.partition_C(gO);
    auto t = make_tensor_like<OutType>(tCrC);
    copy(tCrC, t);

    ThrCopy r2g_thr_copy_o = r2g_tiled_copy_o.get_slice(tid);
    Tensor tCrC_r2g = r2g_thr_copy_o.retile_S(t);
    Tensor tCgO_r2g = r2g_thr_copy_o.retile_D(tCgO);
    copy(r2g_tiled_copy_o, tCrC_r2g, tCgO_r2g);
  }
}

template <int M, int N, int K, typename OutType, typename ComputeTypeA, typename ComputeTypeB, typename ComputeTypeC = OutType>
torch::Tensor run_tiled_copy(const torch::Tensor a, const torch::Tensor b, std::optional<torch::Tensor> _c) {
  at::cuda::CUDAGuard device_guard(a.get_device());
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  auto torch_compute_type_a = to_torch_scalar_type<ComputeTypeA>();
  auto torch_compute_type_b = to_torch_scalar_type<ComputeTypeB>();
  auto torch_compute_type_c = to_torch_scalar_type<ComputeTypeC>();

  torch::Tensor c, out;
  bool is_gemm;

  if (!_c.has_value()){
    auto options = torch::TensorOptions().dtype(torch_compute_type_c).device(torch::kCUDA);
    c = torch::empty({M, N}, options);
    is_gemm = true;
  }
  else {
    c = _c.value();
    is_gemm = false;
  }

  CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a)
  CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type_b)
  CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type_c)

  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, N, K)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)

  constexpr bool isCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();

  if constexpr (isCvtPrecision){
    auto torch_compute_type_out = to_torch_scalar_type<OutType>();
    auto options = torch::TensorOptions().dtype(torch_compute_type_out).device(torch::kCUDA);
    out = torch::empty({M, N}, options);

    CHECK_TORCH_TENSOR_DTYPE(out, torch_compute_type_out)
    CHECK_TORCH_TENSOR_SHAPE(out, M, N)
  }

  using Spec = spec::KernelSpec<OutType, ComputeTypeA, ComputeTypeB, ComputeTypeC, M, N, K>;
  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaDeviceSynchronize();
  auto get_data_ptr = [](const torch::Tensor &tensor) -> void * {
    return tensor.defined() ? tensor.data_ptr() : nullptr;
  };
  void *out_ptr = get_data_ptr(out);

  BOOL_SWITCH(is_gemm, IsGemm, [&] {
    tiled_copy<Spec, IsGemm, isCvtPrecision>
        <<<grid, block, shm_size, stream>>>(c.data_ptr(), a.data_ptr(), b.data_ptr(), M, N, K, out_ptr);
  });

  cudaDeviceSynchronize();

  if constexpr (isCvtPrecision)
    return out;
  else
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tiled_copy_bf16_bf16_bf16_fp32",
        &(run_tiled_copy<32, 32, 16, cute::bfloat16_t, cute::bfloat16_t, cute::bfloat16_t, float>),
        "Run a mixed-precision bf16 32x32x16 MMA operation.");
}