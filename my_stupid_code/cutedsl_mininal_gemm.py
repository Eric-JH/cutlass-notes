import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass.cute.runtime import from_dlpack, make_fake_stream

M = 16  
N = 8
K = 8

def make_cute_tensor(t: torch.Tensor) -> cute.Tensor:
    """Convert a torch tensor to a CuTe DSL tensor for kernel input.

    Marks the last (contiguous) dim as the leading dim so CuTe DSL knows the
    strides at compile time without baking the full static shape in.
    """
    return from_dlpack(t, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)

@cute.kernel
def minimal_gemm_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    tiled_mma: cute.TiledMma,
    is_gemm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()

    gA = cute.local_tile(mA, tiler=(M,K), coord=(0, 0))
    gB = cute.local_tile(mB, tiler=(N,K), coord=(0, 0))
    gC = cute.local_tile(mC, tiler=(M,N), coord=(0, 0))

    thr_mma = tiled_mma.get_slice(tid)
    tCgA = thr_mma.partition_A(gA)
    tCgB = thr_mma.partition_B(gB)
    tCgC = thr_mma.partition_C(gC)

    tCrA = thr_mma.make_fragment_A(tCgA)    
    tCrB = thr_mma.make_fragment_B(tCgB)    
    tCrC = thr_mma.make_fragment_C(tCgC)    

    cute.autovec_copy(tCgA, tCrA)
    cute.autovec_copy(tCgB, tCrB)

    if cutlass.const_expr(is_gemm):
        tCrC.fill(0.0)
    else:
        cute.autovec_copy(tCgC, tCrC)

    cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    cute.autovec_copy(tCrC, tCrC)


@cute.jit
def minimal_gemm(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    stream: CUstream,
    is_gemm: cutlass.Constexpr[bool],
):
    op = cute.nvgpu.warp.MmaF16BF16Op(
        cutlass.Float16,
        cutlass.Float16,
        (M, N, K),
    )

    tiled_mma = cute.make_tiled_mma(op)

    num_threads = tiled_mma.size

    minimal_gemm_kernel(mA, mB, mC, tiled_mma, is_gemm).launch(
        grid=(1, 1, 1),
        block=(num_threads, 1, 1),
        stream=stream,
    )

def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable GPU.")

    Ms = [M]
    Ns = [N]
    Ks = [K]
    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

    a = torch.empty(M, K, device="cuda", dtype=torch.half)
    b = torch.empty(N, K, device="cuda", dtype=torch.half)
    c = torch.empty(M, N, device="cuda", dtype=torch.half)

    gemm_clear = cute.compile(
        minimal_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        True,
        options="--enable-tvm-ffi --generate-line-info",
    )

    gemm_accum = cute.compile(
        minimal_gemm,
        make_cute_tensor(a),
        make_cute_tensor(b),
        make_cute_tensor(c),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        False,
        options="--enable-tvm-ffi --generate-line-info",
    )

    for exp in exps:
        m, n, k = exp

        a = torch.randn(m, k, device="cuda", dtype=torch.half)
        b = torch.randn(n, k, device="cuda", dtype=torch.half)
        c = torch.randn(m, n, device="cuda", dtype=torch.half)

        c_out = torch.empty(m, n, device="cuda", dtype=torch.half)
        gemm_clear(a, b, c_out)
        torch.cuda.synchronize()

        c_inout = c.clone()
        gemm_accum(a, b, c_inout)
        torch.cuda.synchronize()

    print(f"finished")

if __name__ == "__main__":
    main()