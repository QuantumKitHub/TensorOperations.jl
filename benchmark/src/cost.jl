# Analytical flop/byte counts computed from a spec alone, so timings can be reported as
# GFLOP/s / GB/s scaling curves.

# `nothing` eltype means "provider not chosen yet"; assume 8 bytes (never overestimates memory).
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

"""
    intensity(spec::AbstractCaseSpec) -> Float64

Arithmetic intensity `flops(spec) / bytes(spec)`, in flops/byte. Low intensity (permutations,
outer-product-free traces) is memory-bound; high intensity (GEMM-like contractions) is
compute-bound -- useful for separating "this got slower because of bandwidth" from "this got
slower because of compute" across the case set.
"""
intensity(spec::AbstractCaseSpec) = flops(spec) / bytes(spec)

"""
    isblasequivalent(spec::AbstractCaseSpec) -> Bool

Whether `spec` can be executed as a single BLAS call after only a reshape, with no data
permutation: an identity `AddSpec`, or a `ContractSpec` whose contracted indices form a
contiguous block at the tail of `IA` and the head of `IB`, in the same relative order in both
(so a single flatten of the shared dimension is consistent between operands -- a `ContractSpec`
where the contracted indices appear in a *different* relative order in `IA` vs `IB`, e.g. TAPP's
`D[a,d,e] = A[a,b,c]*B[c,d,e,b]`, is contiguous in both operands yet still not BLAS-equivalent).
Used to auto-tag cases `"blas"` (see registry.jl); a `BatchedContractSpec` is never
BLAS-equivalent regardless of its per-slice layout, since it has no fused batched-GEMM
primitive to call as one BLAS operation.
"""
isblasequivalent(::AbstractCaseSpec) = false
isblasequivalent(spec::AddSpec) = spec.IA == spec.IC
function isblasequivalent(spec::ContractSpec)
    contractedA = intersect(spec.IA, spec.IB)
    contractedB = intersect(spec.IB, spec.IA)
    contractedA == contractedB || return false
    openA = setdiff(spec.IA, contractedA)
    openB = setdiff(spec.IB, contractedB)
    return spec.IA == vcat(openA, contractedA) && spec.IB == vcat(contractedB, openB)
end

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

# standard GEMM-equivalent flop count: 2 * open-A * open-B * contracted
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

# batch independent tensorcontract! calls: batch x per-slice cost, not one bigger contraction.
function flops(spec::BatchedContractSpec)
    contracted = intersect(spec.IA, spec.IB)
    openA = setdiff(spec.IA, contracted)
    openB = setdiff(spec.IB, contracted)
    nopenA = prod((spec.dims[l] for l in openA); init = 1)
    nopenB = prod((spec.dims[l] for l in openB); init = 1)
    ncontracted = prod((spec.dims[l] for l in contracted); init = 1)
    return spec.batch * 2 * nopenA * nopenB * ncontracted
end
function bytes(spec::BatchedContractSpec)
    nA = prod((spec.dims[l] for l in spec.IA); init = 1)
    nB = prod((spec.dims[l] for l in spec.IB); init = 1)
    nC = prod((spec.dims[l] for l in spec.IC); init = 1)
    return spec.batch * (nA * _elsize(spec.TA) + nB * _elsize(spec.TB) + nC * _elsize(spec.TC))
end

# walks the same tree ncon itself builds (ncontree/indexordertree), not an arbitrary pairing.
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

# tree: leaves are Int indices into spec.indexlists, nodes are Any[left, right] (ncontree's format).
function _tree_cost(f, spec::NetworkSpec, tree)
    tree isa Int && return spec.indexlists[tree], 0
    labelsA, costA = _tree_cost(f, spec, tree[1])
    labelsB, costB = _tree_cost(f, spec, tree[2])
    contracted = intersect(labelsA, labelsB)
    return symdiff(labelsA, labelsB), costA + costB + f(labelsA, labelsB, contracted)
end
