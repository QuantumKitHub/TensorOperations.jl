module TensorOperationsBenchmarks

using LinearAlgebra: BLAS
using Strided: Strided
using Random: Random, randn!
using BenchmarkTools
using TensorOperations
using TensorOperations: DefaultBackend, DefaultAllocator, AbstractBackend

include("specs.jl")
include("cost.jl")
include("provider.jl")
include("threading.jl")
include("registry.jl")
include("lowering.jl")
include("suite.jl")
include("report.jl")

include("categories/tccg.jl")       # defines _tccg_cases, merged into :pairwise below
include("categories/pairwise.jl")
include("categories/permute.jl")
include("categories/trace.jl")
include("categories/mixed_precision.jl")
include("categories/mps.jl")
include("categories/ctmrg.jl")
include("categories/trg.jl")

export AbstractCaseSpec, AddSpec, TraceSpec, ContractSpec, NetworkSpec
export flops, bytes
export AbstractProvider, ArrayProvider, scalartype, randtensor, backend, allocator, label,
    supports, rng
export ThreadConfig, with_threads, set_threads!
export BenchmarkCase, register_category!, REGISTRY, default_sizes, casetags
export build_suite
export resultstable
export @tagged

end # module
