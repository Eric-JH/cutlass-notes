import os

import torch
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_dir = os.path.join(current_dir, "../third-party/cutlass/include")
source = os.path.join(current_dir, "minimal_gemm.cu")

os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(map(str, torch.cuda.get_device_capability()))
print(f"TORCH_CUDA_ARCH_LIST = {os.environ['TORCH_CUDA_ARCH_LIST']}")

lib = load(
    name = "minimal_gemm",
    sources = source,
    extra_cuda_cflags = [
        "-O3",
        f"-I{cutlass_dir}",
        "-U__CUDA__NO_HALF_OPERATORS__",
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
    ],
    extra_cflags = ["-std=c++17"],
    verbose = True,
)

ENABLE_PROF = os.environ.get("ENABLE_PROF", False)
PRINT_LENGTH = 100

Ms = [16]
Ns = [8]
Ks = [8]
exps = [(m, n, k) for m in Ms for n in Ns for k in Ks]

torch.cuda.manual_seed_all(9527)

for exp in exps:
    M, N, K = exp
    print(f" M={M}, N={N}, K={K} ".center(PRINT_LENGTH, "-"))
    
    a = torch.randn(M, K, device="cuda", dtype=torch.half)
    b = torch.randn(K, N, device="cuda", dtype=torch.half)
    c = torch.randn(M, N, device="cuda", dtype=torch.half)
    
    lib.minimal_gemm(a, b, c)
    c = c.to("cpu")
    c = c.float()