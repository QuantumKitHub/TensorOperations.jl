# PkgBenchmark entrypoint: expects a top-level `const SUITE`. Thread counts come from env vars
# (set by scripts/run_benchmarks.jl) and are applied once, before SUITE is built.
using TensorOperationsBenchmarks
using TensorOperations: StridedNative, StridedBLAS

set_threads!(
    ThreadConfig(;
        blas = tryparse(Int, get(ENV, "TOB_BLAS_THREADS", "")),
        strided = tryparse(Int, get(ENV, "TOB_STRIDED_THREADS", "")),
    )
)

const ELTYPES = (Float64, ComplexF64)
const BACKENDS = (StridedNative(), StridedBLAS())
const PROVIDERS = [ArrayProvider{T}(; backend) for T in ELTYPES for backend in BACKENDS]

const SUITE = build_suite(PROVIDERS)
