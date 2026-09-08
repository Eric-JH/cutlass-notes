import os

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include_path = os.path.join(current_dir, "../third-party/cutlass/include")
source = os.path.join(current_dir, "conv2d.cu")

os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, torch.cuda.get_device_capability()))

lib = load(
    name="conv2d",
    sources=[source],
    extra_cuda_cflags=[
        "-O3",
        f"-I{cutlass_include_path}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ftemplate-backtrace-limit=0",
        "--resource-usage",
        "--generate-line-info",
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",
        "-DCUTLASS_ENABLE_GDC_FOR_SM100",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "-DNDEBUG",
        "-Xfatbin",
        "-compress-all",
    ],
    extra_cflags=["-std=c++17"],
    verbose=True,
)


PRINT_LENGTH = 100


def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    diff = target - ref
    return (torch.norm(diff, p=2) / (torch.norm(ref, p=2) + eps)).item()


def compare(kernel_output: torch.Tensor, torch_output: torch.Tensor):
    kernel_output = kernel_output.float()
    torch_output = torch_output.float()

    max_diff = torch.max(torch.abs(torch_output - kernel_output)).item()
    mean_diff = torch.mean(torch.abs(torch_output - kernel_output)).item()
    re = relative_error(kernel_output, torch_output)
    exact = (kernel_output == torch_output).float().mean().item()
    is_correct = re < 1e-3

    print(
        f" Result: {'Success' if is_correct else 'Failed'}, "
        f"Max diff = {max_diff:.5f}, Mean diff = {mean_diff:.5f}, "
        f"RE = {(re * 100):.4f}%, exact = {exact * 100:.2f}% ".center(PRINT_LENGTH, "-")
    )
    return is_correct


def benchmark(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def reference_conv2d(x: torch.Tensor, w: torch.Tensor, pad: int) -> torch.Tensor:
    # x: (H, W, C) NHWC, w: (N, KH, KW, C) -> (H, W, N) NHWC.
    x_nchw = x.double().permute(2, 0, 1).unsqueeze(0)
    w_nchw = w.double().permute(0, 3, 1, 2)
    return F.conv2d(x_nchw, w_nchw, bias=None, stride=1, padding=pad)[0].permute(1, 2, 0).half()


def main():
    torch.manual_seed(9527)
    torch.cuda.manual_seed_all(9527)

    H, W, C = 540, 960, 4
    KH = KW = 3
    pad = 1

    x = torch.randn(H, W, C, device="cuda", dtype=torch.half)
    print(f" x: {tuple(x.shape)} {x.dtype} ".center(PRINT_LENGTH, "-"))

    ok = True
    for N in [16, 32]:
        w = torch.randn(N, KH, KW, C, device="cuda", dtype=torch.half)
        ref = reference_conv2d(x, w, pad)
        print(f" N = {N} | w: {tuple(w.shape)} {w.dtype} ".center(PRINT_LENGTH, "-"))

        for name in ["conv2d", "conv2d_f16acc"]:
            out = getattr(lib, name)(x, w)
            assert out.shape == (H, W, N), out.shape
            ok &= compare(out, ref)
            ms = benchmark(lambda: getattr(lib, name)(x, w))
            gbs = H * W * N * 2 / ms / 1e6
            print(f" {name}: {ms:.4f} ms | {gbs:.2f} GB/s (output) ".center(PRINT_LENGTH, "-"))

    print(f" Summary: {'all passed' if ok else 'FAILED'} ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
