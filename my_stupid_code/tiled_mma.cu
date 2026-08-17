#include "helper.h"


namespace spec {
    using namespace cute;

    template <typename OutType_,
            typename ComputeTypeA_,
            typename ComputeTypeB_,
            typename ComputeTypeC_,
            int kTileM_ = 32,
            int kTileN_ = 32,
            int kTileK_ = 16>
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
        using TileMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));

        static constexpr int kThreadNum = size(TileMMA{});
        static constexpr int kShmSize = 0;
    };
}

template <typename Spec, bool IsGemm, bool IsCvtPrecision>
__global__ void tiled_mma(void *Cptr, const void *Aptr, const void *Bptr, int M, int N, int K, void *Outptr) {
    using namespace cute;

    using X = Underscore;
    using OutType = Spec::OutType;
    using ComputeTypeA = Spec::ComputeTypeA;
    using ComputeTypeB = Spec::ComputeTypeB;
    using ComputeTypeC = Spec::ComputeTypeC;
    using TileMMA = typename Spec::TileMMA;

    static constexpr int kTileM = Spec::kTileM;
    static constexpr int kTileN = Spec::kTileN;
    static constexpr int kTileK = Spec::kTileK;

    int tid = threadIdx.x;

    Tensor mA = make_tensor(make_gmem_ptr((ComputeTypeA *)Aptr), make_shape(M, K), make_stride(K, Int<1>{})); // (M, K)
    Tensor mB = make_tensor(make_gmem_ptr((ComputeTypeB *)Bptr), make_shape(N, K), make_stride(K, Int<1>{})); // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr((ComputeTypeC *)Cptr), make_shape(M, N), make_stride(N, Int<1>{})); // (M, N)
    Tensor mO = make_tensor(make_gmem_ptr((OutType *)Outptr), make_shape(M, N), make_stride(N, Int<1>{}));    // (M, N)

    auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
    auto coord = make_coord(0, 0, 0);

    Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});
    Tensor gO = local_tile(mO, tiler, coord, Step<_1, _1, X>{});

    TileMMA tiled_mma;
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

    if constexpr (!IsCvtPrecision){
        copy(copy_atom, tCrC, tCgC);
    }
    else {
        Tensor tCgO = thr_mma.partition_C(gO);
        auto t = make_tensor_like<OutType>(tCrC);
        copy(tCrC, t); // Convert precision
        copy(copy_atom, t, tCgO);
    }
}

template <typename ComputeTypeA, 
            typename ComputeTypeB, 
            typename ComputeTypeC, 
            typename OutType, 
            int M, 
            int N, 
            int K>
torch::Tensor run_tiled_mma(const torch::Tensor &a, const torch::Tensor &b, std::optional<torch::Tensor> _c){

    at::cuda::CUDAGuard device_guard(a.get_device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto torch_compute_type_a = to_torch_scalar_type<ComputeTypeA>();
    auto torch_compute_type_b = to_torch_scalar_type<ComputeTypeB>();
    auto torch_compute_type_c = to_torch_scalar_type<ComputeTypeC>();

    torch::Tensor c, out;
    bool is_gemm;

    if (!_c.has_value()) {
        auto options = torch::TensorOptions().dtype(torch_compute_type_c).device(torch::kCUDA);
        c = torch::empty({M, N}, options);
        is_gemm = true;
    }
    else {
        c = _c.value();
        is_gemm = false;
    }

    CHECK_TORCH_TENSOR_DTYPE(a, torch_compute_type_a);
    CHECK_TORCH_TENSOR_DTYPE(b, torch_compute_type_b);
    CHECK_TORCH_TENSOR_DTYPE(c, torch_compute_type_c);
    
    CHECK_TORCH_TENSOR_SHAPE(a, M, K);
    CHECK_TORCH_TENSOR_SHAPE(b, N, K);
    CHECK_TORCH_TENSOR_SHAPE(c, M, N);

    constexpr bool IsCvtPrecision = needs_precision_conversion<ComputeTypeC, OutType>();
    if constexpr (IsCvtPrecision) {
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
    
    auto get_data_ptr = [](const torch::Tensor &tensor) -> void * {
        return tensor.defined() ? tensor.data_ptr() : nullptr;
    };
    void *out_ptr = get_data_ptr(out);

    cudaDeviceSynchronize();

    BOOL_SWITCH(is_gemm, IsGemm, [&] {
        tiled_mma<Spec, IsGemm, IsCvtPrecision>
            <<<grid, block, shm_size, stream>>>(c.data_ptr(), a.data_ptr(), b.data_ptr(), M, N, K, out_ptr);
    });
    cudaDeviceSynchronize();

    if constexpr (IsCvtPrecision)
        return out;
    else
        return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_mma_bf16_bf16_bf16_fp32", 
        &(run_tiled_mma<cute::bfloat16_t, cute::bfloat16_t, float, cute::bfloat16_t, 32, 32, 16>), "Run a single 16x8x8 MMA operation.");
}
