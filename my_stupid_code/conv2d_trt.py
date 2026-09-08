import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    import tensorrt as trt
except ImportError:
    import tensorrt_bindings as trt


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from conv2d import lib  # noqa: E402  (builds / reuses the CuTe CUDA extension)


PRINT_LENGTH = 100
H, W, C, N = 540, 960, 4, 16
KH = KW = 3
PAD = 1


def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    diff = target - ref
    return (torch.norm(diff, p=2) / (torch.norm(ref, p=2) + eps)).item()


def compare(name: str, out: torch.Tensor, ref: torch.Tensor):
    out_f = out.float()
    ref_f = ref.float()
    max_diff = torch.max(torch.abs(ref_f - out_f)).item()
    re = relative_error(out_f, ref_f)
    exact = (out_f == ref_f).float().mean().item()
    ok = re < 1e-3
    print(
        f" {name}: {'OK ' if ok else 'FAIL'} | max={max_diff:.5f} mean={torch.mean(torch.abs(ref_f - out_f)).item():.5f} "
        f"RE={re * 100:.4f}% exact={exact * 100:.2f}% ".center(PRINT_LENGTH, "-")
    )
    return ok


def bench(fn, warmup: int = 20, iters: int = 100):
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


def reference_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x_nchw = x.double().permute(2, 0, 1).unsqueeze(0)
    w_nchw = w.double().permute(0, 3, 1, 2)
    return F.conv2d(x_nchw, w_nchw, bias=None, stride=1, padding=PAD)[0].permute(1, 2, 0).half()


def build_engine(w_nchw: np.ndarray, io_format: str) -> "trt.ICudaEngine":
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        | 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    # Strongly typed network: layer precision is fixed by the tensor dtypes, so
    # BuilderFlag.FP16 must not be set (TensorRT rejects it).
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    inp = network.add_input("x", trt.float16, (1, C, H, W))
    if io_format == "hwc8":
        # Channel-last FP16 boundary.  C is zero-padded to 8 in the buffer, which
        # matches TensorRT's native NHWC tensor-core path.
        inp.allowed_formats = 1 << int(trt.TensorFormat.HWC8)

    conv = network.add_convolution_nd(inp, N, (KH, KW), trt.Weights(w_nchw), trt.Weights())
    conv.stride_nd = (1, 1)
    conv.padding_nd = (PAD, PAD)

    out = conv.get_output(0)
    out.name = "y"
    network.mark_output(out)
    if io_format == "hwc8":
        # allowed_formats only takes effect on network I/O tensors.
        out.allowed_formats = 1 << int(trt.TensorFormat.HWC8)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build the {io_format} engine")
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError(f"TensorRT failed to deserialize the {io_format} engine")
    return engine


class TRTRunner:
    def __init__(self, engine: "trt.ICudaEngine", input_shape, stream: int):
        self.context = engine.create_execution_context()
        self.context.set_input_shape("x", input_shape)
        self.stream = stream
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(
                f"   [trt] {name}: shape={tuple(engine.get_tensor_shape(name))} "
                f"format={engine.get_tensor_format(name)} dtype={engine.get_tensor_dtype(name)}"
            )

    def __call__(self, x_ptr: int, y_ptr: int):
        self.context.set_tensor_address("x", x_ptr)
        self.context.set_tensor_address("y", y_ptr)
        self.context.execute_async_v3(self.stream)


def main():
    torch.manual_seed(9527)
    torch.cuda.manual_seed_all(9527)

    x = torch.randn(H, W, C, device="cuda", dtype=torch.half)
    w = torch.randn(N, KH, KW, C, device="cuda", dtype=torch.half)
    ref = reference_conv2d(x, w)
    w_nchw = w.permute(0, 3, 1, 2).contiguous().cpu().numpy()

    print(f" x: {tuple(x.shape)} {x.dtype} | w: {tuple(w.shape)} {w.dtype} ".center(PRINT_LENGTH, "-"))

    ok = True

    # ---------------- our CuTe kernels (NHWC in / NHWC out) ----------------
    y_ours = lib.conv2d(x, w)
    ok &= compare("CuTe kernel (fp32 acc)", y_ours, ref)
    y_ours_f16 = lib.conv2d_f16acc(x, w)
    ok &= compare("CuTe kernel (fp16 acc)", y_ours_f16, ref)

    # ---------------- TensorRT, NCHW boundary ----------------
    print(" Building TensorRT NCHW engine ...".center(PRINT_LENGTH, "-"))
    trt_stream = torch.cuda.Stream()
    engine_nchw = build_engine(w_nchw, "nchw")
    runner_nchw = TRTRunner(engine_nchw, (1, C, H, W), trt_stream.cuda_stream)

    x_nchw = x.permute(2, 0, 1).contiguous().unsqueeze(0)
    y_nchw = torch.empty((1, N, H, W), device="cuda", dtype=torch.half)
    runner_nchw(x_nchw.data_ptr(), y_nchw.data_ptr())
    torch.cuda.synchronize()
    y_trt_nchw = y_nchw[0].permute(1, 2, 0)
    ok &= compare("TensorRT NCHW (kernel only)", y_trt_nchw, ref)

    # ---------------- TensorRT, HWC8 channel-last boundary ----------------
    print(" Building TensorRT HWC8 engine ...".center(PRINT_LENGTH, "-"))
    engine_hwc8 = build_engine(w_nchw, "hwc8")
    runner_hwc8 = TRTRunner(engine_hwc8, (1, C, H, W), trt_stream.cuda_stream)

    x_hwc8 = torch.zeros((1, H, W, 8), device="cuda", dtype=torch.half)
    x_hwc8[0, :, :, :C] = x
    y_hwc8 = torch.empty((1, H, W, N), device="cuda", dtype=torch.half)
    runner_hwc8(x_hwc8.data_ptr(), y_hwc8.data_ptr())
    torch.cuda.synchronize()
    ok &= compare("TensorRT HWC8 (kernel only)", y_hwc8[0], ref)

    # ---------------- timing ----------------
    print(" Latency (warmup=20, iters=100) ".center(PRINT_LENGTH, "-"))

    with torch.cuda.stream(trt_stream):
        ms_ours = bench(lambda: lib.conv2d(x, w))
        ms_ours_f16 = bench(lambda: lib.conv2d_f16acc(x, w))
        ms_nchw = bench(lambda: runner_nchw(x_nchw.data_ptr(), y_nchw.data_ptr()))
        ms_hwc8 = bench(lambda: runner_hwc8(x_hwc8.data_ptr(), y_hwc8.data_ptr()))

    # End-to-end TRT NCHW: include NHWC -> NCHW packing and NCHW -> NHWC unpacking.
    x_nchw_buf = torch.empty((1, C, H, W), device="cuda", dtype=torch.half)
    y_nchw_out = torch.empty((H, W, N), device="cuda", dtype=torch.half)

    def trt_nchw_e2e():
        x_nchw_buf[0].copy_(x.permute(2, 0, 1))
        runner_nchw(x_nchw_buf.data_ptr(), y_nchw.data_ptr())
        y_nchw_out.copy_(y_nchw[0].permute(1, 2, 0))

    with torch.cuda.stream(trt_stream):
        ms_nchw_e2e = bench(trt_nchw_e2e)

    rows = [
        ("CuTe kernel (NHWC, fp32 acc)", ms_ours),
        ("CuTe kernel (NHWC, fp16 acc)", ms_ours_f16),
        ("TensorRT NCHW (kernel only)", ms_nchw),
        ("TensorRT HWC8 (kernel only)", ms_hwc8),
        ("TensorRT NCHW (incl. layout conv)", ms_nchw_e2e),
    ]
    best = min(r[1] for r in rows)
    for name, ms in rows:
        gbs = H * W * N * 2 / ms / 1e6
        print(f"   {name:<36} {ms:7.4f} ms  {gbs:7.1f} GB/s  ({ms / best:.2f}x) ".center(PRINT_LENGTH, "-"))

    print(f" Summary: {'all outputs correct' if ok else 'MISMATCH'} ".center(PRINT_LENGTH, "-"))


if __name__ == "__main__":
    main()
