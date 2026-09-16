module TensorOperationsBenchmarks

using LinearAlgebra: BLAS
using Strided: Strided
using Random: Random, randn!
using DataFrames: DataFrame
using BenchmarkTools
using TensorOperations
using TensorOperations: DefaultBackend, DefaultAllocator, AbstractBackend

# Specs (pure data) and the cost model computed from them.
include("specs.jl")
include("cost.jl")

# The downstream extension points: what tensor type to run against, and how many threads.
include("provider.jl")
include("threading.jl")

# The case registry, and turning a (spec, provider) pair into an executable benchmark.
include("registry.jl")
include("lowering.jl")

# Suite assembly and reporting.
include("suite.jl")
include("report.jl")

# Categories: tccg.jl/mps.jl/ctmrg.jl/trg.jl define generators merged by contract.jl/network.jl
# (include order doesn't matter -- generators are only called after the whole module loads).
include("categories/tccg.jl")
include("categories/contract.jl")
include("categories/permute.jl")
include("categories/trace.jl")
include("categories/mixed_precision.jl")
include("categories/mps.jl")
include("categories/ctmrg.jl")
include("categories/trg.jl")
include("categories/network.jl")

export AbstractCaseSpec, AddSpec, TraceSpec, ContractSpec, NetworkSpec
export flops, bytes
export AbstractProvider, ArrayProvider, scalartype, randtensor, backend, allocator, label,
    supports, rng
export ThreadConfig, with_threads, set_threads!
export BenchmarkCase, register_category!, REGISTRY, default_sizes, within_memory_budget
export build_suite
export resultstable
export @tagged

end # module
