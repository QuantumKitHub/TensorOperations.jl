# Pure-data contraction/network specs: labels and leg extents only, no tensor objects, so a
# spec is shareable across providers. `TA`/`TB`/`TC`/`Ts` default to `nothing` ("provider's
# default scalartype"); only mixed_precision.jl sets them explicitly.

abstract type AbstractCaseSpec end

"""
    AddSpec(IA, IC, dims, conjA, TA=nothing, TC=nothing)

Specification of a permutation `C = permutedims(opA(A), pA)` (executed via `tensorcopy`),
where `pA` is derived from matching labels in `IA` to `IC`.
"""
struct AddSpec <: AbstractCaseSpec
    IA::Vector{Symbol}
    IC::Vector{Symbol}
    dims::Dict{Symbol, Int}
    conjA::Bool
    TA::Union{Nothing, Type}
    TC::Union{Nothing, Type}
end
function AddSpec(IA, IC, dims; conjA::Bool = false, TA = nothing, TC = nothing)
    return AddSpec(collect(Symbol, IA), collect(Symbol, IC), dims, conjA, TA, TC)
end

"""
    TraceSpec(IA, IC, dims, conjA, TA=nothing, TC=nothing)

Specification of a (partial) trace `C = permutedims(trace(opA(A)), p)` (executed via
`tensortrace`). Labels appearing twice in `IA` are traced; labels appearing once and also
in `IC` are kept.
"""
struct TraceSpec <: AbstractCaseSpec
    IA::Vector{Symbol}
    IC::Vector{Symbol}
    dims::Dict{Symbol, Int}
    conjA::Bool
    TA::Union{Nothing, Type}
    TC::Union{Nothing, Type}
end
function TraceSpec(IA, IC, dims; conjA::Bool = false, TA = nothing, TC = nothing)
    return TraceSpec(collect(Symbol, IA), collect(Symbol, IC), dims, conjA, TA, TC)
end

"""
    ContractSpec(IA, IB, IC, dims, conjA, conjB, TA=nothing, TB=nothing, TC=nothing)

Specification of a pairwise contraction `C = contract(opA(A), opB(B))` (executed via
`tensorcontract`).
"""
struct ContractSpec <: AbstractCaseSpec
    IA::Vector{Symbol}
    IB::Vector{Symbol}
    IC::Vector{Symbol}
    dims::Dict{Symbol, Int}
    conjA::Bool
    conjB::Bool
    TA::Union{Nothing, Type}
    TB::Union{Nothing, Type}
    TC::Union{Nothing, Type}
end
function ContractSpec(
        IA, IB, IC, dims; conjA::Bool = false, conjB::Bool = false,
        TA = nothing, TB = nothing, TC = nothing
    )
    return ContractSpec(
        collect(Symbol, IA), collect(Symbol, IB), collect(Symbol, IC), dims,
        conjA, conjB, TA, TB, TC
    )
end

"""
    BatchedContractSpec(batch, IA, IB, IC, dims, conjA=false, conjB=false, TA=nothing, TB=nothing, TC=nothing)

`batch` independent pairwise contractions, each with the same label structure, executed as
`batch` separate `tensorcontract!` calls -- TensorOperations has no fused batched-GEMM
primitive, so this genuinely cannot collapse to one BLAS call, unlike a `ContractSpec` with a
`:gemm_ready` layout. Models a "many small contractions" regime (e.g. attention-style batched
matmuls) where per-call dispatch overhead dominates rather than raw FLOPs.
"""
struct BatchedContractSpec <: AbstractCaseSpec
    batch::Int
    IA::Vector{Symbol}
    IB::Vector{Symbol}
    IC::Vector{Symbol}
    dims::Dict{Symbol, Int}
    conjA::Bool
    conjB::Bool
    TA::Union{Nothing, Type}
    TB::Union{Nothing, Type}
    TC::Union{Nothing, Type}
end
function BatchedContractSpec(
        batch, IA, IB, IC, dims; conjA::Bool = false, conjB::Bool = false,
        TA = nothing, TB = nothing, TC = nothing
    )
    return BatchedContractSpec(
        Int(batch), collect(Symbol, IA), collect(Symbol, IB), collect(Symbol, IC), dims,
        conjA, conjB, TA, TB, TC
    )
end

"""
    NetworkSpec(indexlists, conjlist, output, dims, order=nothing, Ts=nothing)

An `ncon`-style multi-tensor network: `indexlists[k]` gives the signed integer index labels
of the `k`th tensor (positive = contracted, negative = open/output), `dims` maps each *label*
(by absolute value) to its extent, and `order` optionally fixes the contraction order (as a
list of positive labels, in the order they should be contracted) -- `nothing` lets `ncon`'s
default greedy tree builder decide.
"""
struct NetworkSpec <: AbstractCaseSpec
    indexlists::Vector{Vector{Int}}
    conjlist::Vector{Bool}
    output::Vector{Int}
    dims::Dict{Int, Int}
    order::Union{Nothing, Vector{Int}}
    Ts::Union{Nothing, Vector{<:Type}}
end
function NetworkSpec(
        indexlists, dims; conjlist = fill(false, length(indexlists)),
        output = nothing, order = nothing, Ts = nothing
    )
    outputindices = something(
        output, sort(unique(l for il in indexlists for l in il if l < 0); rev = true)
    )
    return NetworkSpec(
        [collect(Int, il) for il in indexlists], collect(Bool, conjlist),
        collect(Int, outputindices), dims, order, Ts
    )
end

# Compact einsum-style show methods, e.g. `ContractSpec: C[i,j] = A[i,k] * B[k,j] (dim=64)`.
_dimsnote(dims::Dict) = (vals = unique(values(dims)); length(vals) == 1 ? " (dim=$(only(vals)))" : " (dims=$dims)")
_opstr(label, conj) = conj ? "conj($label)" : label

function Base.show(io::IO, spec::AddSpec)
    return print(
        io, "AddSpec: C[", join(spec.IC, ","), "] = ",
        _opstr("A[$(join(spec.IA, ","))]", spec.conjA), _dimsnote(spec.dims)
    )
end

function Base.show(io::IO, spec::TraceSpec)
    return print(
        io, "TraceSpec: C[", join(spec.IC, ","), "] = tr(",
        _opstr("A[$(join(spec.IA, ","))]", spec.conjA), ")", _dimsnote(spec.dims)
    )
end

function Base.show(io::IO, spec::ContractSpec)
    return print(
        io, "ContractSpec: C[", join(spec.IC, ","), "] = ",
        _opstr("A[$(join(spec.IA, ","))]", spec.conjA), " * ",
        _opstr("B[$(join(spec.IB, ","))]", spec.conjB), _dimsnote(spec.dims)
    )
end

function Base.show(io::IO, spec::BatchedContractSpec)
    return print(
        io, "BatchedContractSpec: ", spec.batch, " x C[", join(spec.IC, ","), "] = ",
        _opstr("A[$(join(spec.IA, ","))]", spec.conjA), " * ",
        _opstr("B[$(join(spec.IB, ","))]", spec.conjB), _dimsnote(spec.dims)
    )
end

function Base.show(io::IO, spec::NetworkSpec)
    tensors = join(
        (
            _opstr("T$k[$(join(il, ","))]", conj)
                for (k, (il, conj)) in enumerate(zip(spec.indexlists, spec.conjlist))
        ), " * "
    )
    return print(io, "NetworkSpec: C[", join(spec.output, ","), "] = ", tensors, _dimsnote(spec.dims))
end
