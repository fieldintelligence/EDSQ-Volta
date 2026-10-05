import os

from setuptools import setup

# Optional CUDA extension (sm_70 only). Opt-in so a pure-Python install works
# for CI and contract tests, mirroring how 1Cat gates native kernels.
if os.environ.get("VLLM_VOLTA_BUILD_OPS") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    setup(
        ext_modules=[
            CUDAExtension(
                name="vllm_volta._volta_ops",
                sources=["csrc/volta_paged_attention.cu"],
                extra_compile_args={
                    "nvcc": [
                        "-gencode=arch=compute_70,code=sm_70",
                        "-O3",
                        "--use_fast_math",
                    ]
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
else:
    setup()
