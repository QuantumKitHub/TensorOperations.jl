# PkgBenchmark entrypoint: `PkgBenchmark.benchmarkpkg` looks for this file and expects a
# top-level `const SUITE`. Thread counts are read from environment variables and applied via
# `set_threads!` *before* `SUITE` is built -- once, at the process level -- rather than swept as
# a suite axis, so that no timed sample ever pays for a `BLAS`/`Strided` thread-count switch
# (see threading.jl). `scripts/run_benchmarks.jl` sets these env vars per outer sweep point.
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
