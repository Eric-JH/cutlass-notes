import os
import subprocess

import torch
from torch.utils.cpp_extension import load


# --------------------------------------------------------------------------- #
# Build configuration (shared by both extensions)
# --------------------------------------------------------------------------- #

current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include_path = os.path.join(current_dir, "../third-party/cutlass/include")

os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, torch.cuda.get_device_capability()))

# The two .cu files each carry their own PYBIND11_MODULE block, so they cannot
# be compiled into a single extension module (duplicate module symbol). We
# therefore build them as two independent modules that share the same flags.
CUTLASS_CUDA_CFLAGS = [
    "-O3",
    f"-I{cutlass_include_path}",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "--ftemplate-backtrace-limit=0",  # To debug template code
    "--resource-usage",  # printing out number of registers
    # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
    "--generate-line-info",  # show PTX and SASS in ncu
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
    "-DCUTLASS_ENABLE_GDC_FOR_SM100",  # For PDL
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
    "-DNDEBUG",  # Important, otherwise performance is severely impacted
    "-Xfatbin",  # compress all binary sections
    "-compress-all",
    # for debug purpose
    # "-G", # device debug
    # "-g", # host debug
    # "-Xcompiler",
    # "-rdynamic",
]


def build(name, sources):
    """Load a CUDA extension with the shared CUTLASS compile flags."""
    return load(
        name=name,
        sources=[os.path.join(current_dir, filename) for filename in sources],
        extra_cuda_cflags=CUTLASS_CUDA_CFLAGS,
        extra_cflags=["-std=c++17"],
        verbose=True,  # show compile logs
    )


lib_minimal = build("minimal_gemm", ["minimal_gemm.cu"])
lib_mixed = build("mixed_precision_gemm", ["mixed_precision_gemm.cu"])


# --------------------------------------------------------------------------- #
# Comparison utilities (shared across all experiment groups)
# --------------------------------------------------------------------------- #

ENABLE_PROF = os.environ.get("ENABLE_PROF", False)
PRINT_LENGTH = 100


def relative_error(target: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8):
    diff = target - ref
    norm_diff = torch.norm(diff, p=2)
    norm_diff_ref = torch.norm(ref, p=2)

    return (norm_diff / (norm_diff_ref + eps)).item()


num_succeed = 0
num_failed = 0


def compare_matrix(kernel_output: torch.Tensor, torch_output: torch.Tensor):
    kernel_output = kernel_output.float()
    torch_output = torch_output.float()

    max_diff = torch.max(torch.abs(torch_output - kernel_output))
    mean_diff = torch.mean(torch.abs(torch_output - kernel_output))
    re = relative_error(kernel_output, torch_output)
    is_correct = re < 0.001

    global num_succeed, num_failed

    if not is_correct:
        num_failed += 1

        print(f" Kernel Output: {tuple(kernel_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(kernel_output[:8, :8])

        print(f" Torch Output: {tuple(torch_output.shape)} ".center(PRINT_LENGTH, "-"))
        print(torch_output[:8, :8])
    else:
        num_succeed += 1

    print(
        f" Result: {'Success' if is_correct else 'Failed'}, Max diff = {max_diff:.5f}, Mean diff = {mean_diff:.5f}, RE = {(re * 100):.2f}% ".center(
            PRINT_LENGTH, "-"
        )
    )


def run_group(title, Ms, Ns, Ks, make_inputs, kernel_fn, ref_mm, ref_mma):
    """Run one MMA experiment group: a plain matmul (MM) case and an addmm (MMA) case.

    Args:
        title:       Header printed before the group runs.
        Ms, Ns, Ks:  Shape lists; the cartesian product is iterated.
        make_inputs: (M, N, K) -> (a, b, c) tensors for this group.
        kernel_fn:   (a, b, c_or_None) -> Tensor. c=None for MM, c.clone() for MMA.
        ref_mm:      (a, b) -> Tensor. Reference for the plain matmul case.
        ref_mma:     (c, a, b) -> Tensor. Reference for the addmm case.
    """
    print(f"\n {title} ".center(PRINT_LENGTH, "="))

    exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]
    torch.cuda.manual_seed_all(9527)

    for M, N, K in exps:
        print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))

        a, b, c = make_inputs(M, N, K)

        # Case 1: MM (no source accumulator)
        kernel_output = kernel_fn(a, b, None)
        if not ENABLE_PROF:
            compare_matrix(kernel_output, ref_mm(a, b))

        # Case 2: MMA (accumulate into a copy of c)
        kernel_output = kernel_fn(a, b, c.clone())
        if not ENABLE_PROF:
            compare_matrix(kernel_output, ref_mma(c, a, b))


# --------------------------------------------------------------------------- #
# Environment capability check for the fp8 (e4m3 / e5m2) experiment groups
# --------------------------------------------------------------------------- #


def get_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in output.split("\n"):
            if "release" in line:
                return line.split("release ")[1].split(",")[0]
    except Exception as e:
        return f"Error: {e}"


sm_version = torch.cuda.get_device_capability()
cuda_version = tuple(int(k) for k in get_cuda_version().split("."))

# Check if the hardware and software environment supports alternate floating
# point types (e4m3 and e5m2) for MMA operations.
# - SM version 8.9 or higher is required.
# - CUDA version 12.4 or higher (PTX version 8.4 or higher) is required.
fp8_supported = sm_version >= (8, 9) and cuda_version >= (12, 4)


# --------------------------------------------------------------------------- #
# Experiment groups
# --------------------------------------------------------------------------- #

# ---------------- half = half * half + half (minimal_gemm) ----------------
# For fp16 input, `torch.matmul` / `torch.addmm` accumulate in fp16, matching
# the kernel's accumulation order closely enough for the tolerance below.

run_group(
    title="half = half * half + half",
    Ms=[16],
    Ns=[8],
    Ks=[8],
    make_inputs=lambda M, N, K: (
        torch.randn(M, K, device="cuda", dtype=torch.half),
        torch.randn(N, K, device="cuda", dtype=torch.half),
        torch.randn(M, N, device="cuda", dtype=torch.half),
    ),
    kernel_fn=lib_minimal.minimal_gemm,
    ref_mm=lambda a, b: torch.matmul(a, b.T),
    # Mathematically equivalent to matmul(a, b.T) + c, but the accumulation
    # order differs from our kernel, which may cause slight numerical
    # discrepancies due to floating-point arithmetic.
    ref_mma=lambda c, a, b: torch.addmm(c, a, b.T),
)

# ---------------- fp32 = bf16 * bf16 + fp32 ----------------
# For bf16 input, `torch.matmul` uses bf16 as the output precision, which may
# reduce numerical accuracy. We upcast a/b to fp32 before the reference matmul
# to keep higher precision for comparison.

run_group(
    title="fp32 = bf16 * bf16 + fp32",
    Ms=[16],
    Ns=[8],
    Ks=[8],
    make_inputs=lambda M, N, K: (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16),
        torch.randn(N, K, device="cuda", dtype=torch.bfloat16),
        torch.randn(M, N, device="cuda", dtype=torch.float32),
    ),
    kernel_fn=lib_mixed.mixed_precision_gemm_fp32_bf16_bf16_fp32,
    ref_mm=lambda a, b: torch.matmul(a.float(), b.T.float()),
    ref_mma=lambda c, a, b: torch.addmm(c, a.float(), b.T.float()),
)

# ---------------- bf16 = bf16 * bf16 + fp32 ----------------

run_group(
    title="bf16 = bf16 * bf16 + fp32",
    Ms=[16],
    Ns=[8],
    Ks=[8],
    make_inputs=lambda M, N, K: (
        torch.randn(M, K, device="cuda", dtype=torch.bfloat16),
        torch.randn(N, K, device="cuda", dtype=torch.bfloat16),
        torch.randn(M, N, device="cuda", dtype=torch.float32),
    ),
    kernel_fn=lib_mixed.mixed_precision_gemm_bf16_bf16_bf16_fp32,
    # Output is bf16 here, so the plain-MM reference stays in bf16.
    ref_mm=lambda a, b: torch.matmul(a, b.T),
    ref_mma=lambda c, a, b: torch.addmm(c, a.float(), b.T.float()).bfloat16(),
)

# ---------------- fp32 = e4m3 * e5m2 + fp32 ----------------

if fp8_supported:
    run_group(
        title="fp32 = e4m3 * e5m2 + fp32",
        Ms=[16],
        Ns=[8],
        Ks=[32],
        make_inputs=lambda M, N, K: (
            torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn),
            torch.randn(N, K, device="cuda", dtype=torch.float32).to(torch.float8_e5m2),
            torch.randn(M, N, device="cuda", dtype=torch.float32),
        ),
        kernel_fn=lib_mixed.mixed_precision_gemm_fp32_e4m3_e5m2_fp32,
        ref_mm=lambda a, b: torch.matmul(a.float(), b.T.float()),
        ref_mma=lambda c, a, b: torch.addmm(c, a.float(), b.T.float()),
    )

# ---------------- bf16 = e4m3 * e5m2 + fp32 ----------------

if fp8_supported:
    run_group(
        title="bf16 = e4m3 * e5m2 + fp32",
        Ms=[16],
        Ns=[8],
        Ks=[32],
        make_inputs=lambda M, N, K: (
            torch.randn(M, K, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn),
            torch.randn(N, K, device="cuda", dtype=torch.float32).to(torch.float8_e5m2),
            torch.randn(M, N, device="cuda", dtype=torch.float32),
        ),
        kernel_fn=lib_mixed.mixed_precision_gemm_bf16_e4m3_e5m2_fp32,
        ref_mm=lambda a, b: torch.matmul(a.float(), b.T.float()).bfloat16(),
        ref_mma=lambda c, a, b: torch.addmm(c, a.float(), b.T.float()).bfloat16(),
    )


print(f" Summary: {num_succeed} Succeed, {num_failed} Failed ".center(PRINT_LENGTH, "-"))
