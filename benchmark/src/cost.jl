# Analytical flop/byte counts computed directly from a spec (no tensor objects needed), so
# that timings can be reported as GFLOP/s or GB/s scaling curves instead of raw wall-clock time.

# A spec's `TA`/`TB`/`TC` (or `Ts`) are `nothing` unless the mixed-precision category set them
# explicitly -- meaning "whatever the provider's default `scalartype` turns out to be", which
# isn't known until a provider is chosen. For sizing purposes (both for reporting GB/s *before*
# a provider is picked, and for the `within_memory_budget` safety check in registry.jl, which
# runs at case-generation time) we assume the common case of `Float64`/`ComplexF64`-sized
# (8-byte) elements; this can only ever *underestimate* real memory use for a provider using a
# larger element type, never overestimate it into skipping a case that would actually fit.
_elsize(::Nothing) = sizeof(Float64)
_elsize(T::Type) = sizeof(T)

"""
    flops(spec::AbstractCaseSpec) -> Int

Approximate number of floating point operations (multiply + add counted together) needed to
execute `spec`.
"""
function flops end

"""
    bytes(spec::AbstractCaseSpec) -> Int

Approximate number of bytes moved (all tensors read once, output written once) to execute
`spec`.
"""
function bytes end

# Permutation: no arithmetic, just data movement (one read + one write of every element).
flops(::AddSpec) = 0
function bytes(spec::AddSpec)
    n = prod((spec.dims[l] for l in spec.IA); init = 1)
    return n * (_elsize(spec.TA) + _elsize(spec.TC))
end

# Trace: every element of A is read and accumulated once into the (smaller) output.
function flops(spec::TraceSpec)
    return prod((spec.dims[l] for l in spec.IA); init = 1)
end
function bytes(spec::TraceSpec)
    nA = prod((spec.dims[l] for l in spec.IA); init = 1)
    nC = prod((spec.dims[l] for l in spec.IC); init = 1)
    return nA * _elsize(spec.TA) + nC * _elsize(spec.TC)
end

# Pairwise contraction: 2 * (open-A) * (open-B) * (contracted), the standard GEMM-equivalent
# flop count, using multiply-add pairs.
function flops(spec::ContractSpec)
    contracted = intersect(spec.IA, spec.IB)
    openA = setdiff(spec.IA, contracted)
    openB = setdiff(spec.IB, contracted)
    nopenA = prod((spec.dims[l] for l in openA); init = 1)
    nopenB = prod((spec.dims[l] for l in openB); init = 1)
    ncontracted = prod((spec.dims[l] for l in contracted); init = 1)
    return 2 * nopenA * nopenB * ncontracted
end
function bytes(spec::ContractSpec)
    nA = prod((spec.dims[l] for l in spec.IA); init = 1)
    nB = prod((spec.dims[l] for l in spec.IB); init = 1)
    nC = prod((spec.dims[l] for l in spec.IC); init = 1)
    return nA * _elsize(spec.TA) + nB * _elsize(spec.TB) + nC * _elsize(spec.TC)
end

# Network: walk the *actual* pairwise contraction tree that `ncon` would build for this
# network (`TensorOperations.ncontree`/`indexordertree`, the same functions `ncon` itself
# calls), rather than an arbitrary/greedy pairing -- so the reported cost matches what
# `execute(spec::NetworkSpec, ...)` actually runs, not a guess at it.
function flops(spec::NetworkSpec)
    tree = spec.order === nothing ? TensorOperations.ncontree(spec.indexlists) :
        TensorOperations.indexordertree(spec.indexlists, spec.order)
    _, total = _tree_cost(spec, tree) do labelsA, labelsB, contracted
        nopenA = prod((spec.dims[abs(l)] for l in labelsA if !(l in contracted)); init = 1)
        nopenB = prod((spec.dims[abs(l)] for l in labelsB if !(l in contracted)); init = 1)
        ncontracted = prod((spec.dims[abs(l)] for l in contracted); init = 1)
        return 2 * nopenA * nopenB * ncontracted
    end
    return total
end
function bytes(spec::NetworkSpec)
    Ts = something(spec.Ts, fill(nothing, length(spec.indexlists)))
    return sum(
        prod((spec.dims[abs(l)] for l in il); init = 1) * _elsize(T)
            for (il, T) in zip(spec.indexlists, Ts)
    )
end

# Recursively walks a `ncontree`/`indexordertree` result (leaves are `Int` indices into
# `spec.indexlists`, nodes are `Any[left, right]`), returning `(survivinglabels, totalcost)`,
# calling `f(labelsA, labelsB, contractedlabels)` at every pairwise step -- mirroring exactly
# how `ncon`'s own `contracttree` combines subtrees (`IC = symdiff(IA, IB)`).
function _tree_cost(f, spec::NetworkSpec, tree)
    tree isa Int && return spec.indexlists[tree], 0
    labelsA, costA = _tree_cost(f, spec, tree[1])
    labelsB, costB = _tree_cost(f, spec, tree[2])
    contracted = intersect(labelsA, labelsB)
    return symdiff(labelsA, labelsB), costA + costB + f(labelsA, labelsB, contracted)
end
