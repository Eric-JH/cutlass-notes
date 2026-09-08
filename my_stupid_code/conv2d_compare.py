import os
import sys

import torch
import torch.nn.functional as F


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from conv2d import lib  # noqa: E402  (builds / reuses the CUDA extension)


PRINT_LENGTH = 100
H, W, C, N = 540, 960, 4, 16
KH = KW = 3
PAD = 1


def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    return (torch.norm(target - ref, p=2) / (torch.norm(ref, p=2) + eps)).item()


def compare(name: str, out: torch.Tensor, ref: torch.Tensor):
    out_f = out.float()
    ref_f = ref.float()
    max_diff = torch.max(torch.abs(ref_f - out_f)).item()
    re = relative_error(out_f, ref_f)
    ok = re < 1e-3
    print(f" {name}: {'OK ' if ok else 'FAIL'} | max={max_diff:.5f} RE={re * 100:.4f}% ".center(PRINT_LENGTH, "-"))
    return ok


def bench(fn, warmup: int = 50, iters: int = 300):
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


def main():
    torch.manual_seed(9527)
    torch.cuda.manual_seed_all(9527)

    x = torch.randn(H, W, C, device="cuda", dtype=torch.half)
    w = torch.randn(N, KH, KW, C, device="cuda", dtype=torch.half)
    ref = F.conv2d(
        x.double().permute(2, 0, 1).unsqueeze(0),
        w.double().permute(0, 3, 1, 2),
        padding=PAD,
    )[0].permute(1, 2, 0).half()

    print(f" x: {tuple(x.shape)} | w: {tuple(w.shape)} | warmup=50 iters=300 ".center(PRINT_LENGTH, "-"))

    variants = [
        ("materialized im2col (fp32 acc)", "conv2d"),
        ("materialized im2col (fp16 acc)", "conv2d_f16acc"),
        ("implicit GEMM, direct gather (fp32 acc)", "conv2d_implicit"),
        ("implicit GEMM, direct gather (fp16 acc)", "conv2d_implicit_f16acc"),
        ("implicit GEMM, smem patch (fp32 acc)", "conv2d_implicit_smem"),
        ("implicit GEMM, smem patch (fp16 acc)", "conv2d_implicit_smem_f16acc"),
        ("im2col + cp.async pipeline (fp32 acc)", "conv2d_pipeline"),
        ("im2col + cp.async pipeline (fp16 acc)", "conv2d_pipeline_f16acc"),
    ]

    stream = torch.cuda.Stream()
    results = []
    ok = True

    with torch.cuda.stream(stream):
        for name, fn_name in variants:
            fn = getattr(lib, fn_name)
            out = fn(x, w)
            ok &= compare(name, out, ref)
            ms = bench(lambda fn=fn: fn(x, w))
            results.append((name, ms))

    best = min(ms for _, ms in results)
    print(" Latency (warmup=50, iters=300) ".center(PRINT_LENGTH, "-"))
    for name, ms in results:
        gbs = H * W * N * 2 / ms / 1e6
        print(f"   {name:<34} {ms:7.4f} ms  {gbs:7.1f} GB/s  ({ms / best:.2f}x) ".center(PRINT_LENGTH, "-"))

    print(f" Summary: {'all outputs correct' if ok else 'MISMATCH'} ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
