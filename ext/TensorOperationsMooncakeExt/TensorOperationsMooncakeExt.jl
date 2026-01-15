module TensorOperationsMooncakeExt

using TensorOperations
# Mooncake imports ChainRulesCore as CRC to avoid name conflicts
# here we import it ourselves to ensure the rules from the ChainRulesCore
# extension are in fact loaded
using Mooncake, Mooncake.CRC
using TensorOperations: AbstractBackend, DefaultAllocator, CUDAAllocator, ManualAllocator
using TensorOperations: tensoralloc, tensoradd!, tensorcontract!, tensortrace!, _kron, numind, _needs_tangent, numin, numout
using Mooncake: ReverseMode, DefaultCtx, CoDual, NoRData, arrayify, @zero_derivative, primal, tangent
using VectorInterface, TupleTools

Mooncake.tangent_type(::Type{Index2Tuple}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:AbstractBackend}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{DefaultAllocator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{CUDAAllocator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{ManualAllocator}) = Mooncake.NoTangent

trivtuple(N) = ntuple(identity, N)

@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorstructure), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensoradd_structure), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensoradd_type), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensoralloc_add), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorcontract_structure), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensorcontract_type), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.tensoralloc_contract), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.promote_contract), Any}
@zero_derivative DefaultCtx Tuple{typeof(TensorOperations.promote_add), Any}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(TensorOperations.tensorfree!), Any}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{typeof(TensorOperations.tensoralloc), Any, Any, Any, Any}

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(tensorcontract!), AbstractArray, AbstractArray, Index2Tuple, Bool, AbstractArray, Index2Tuple, Bool, Index2Tuple, Number, Number, Vararg{Any}}
function Mooncake.rrule!!(
        ::CoDual{typeof(tensorcontract!)},
        C_dC::CoDual{<:AbstractArray{TC}},
        A_dA::CoDual{<:AbstractArray{TA}},
        pA_dpA::CoDual{<:Index2Tuple},
        conjA_dconjA::CoDual{Bool},
        B_dB::CoDual{<:AbstractArray{TB}},
        pB_dpB::CoDual{<:Index2Tuple},
        conjB_dconjB::CoDual{Bool},
        pAB_dpAB::CoDual{<:Index2Tuple},
        α_dα::CoDual{Tα},
        β_dβ::CoDual{Tβ},
        ba_dba::CoDual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TB <: Number, TC <: Number}
    C, dC = arrayify(C_dC)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    pA = primal(pA_dpA)
    pB = primal(pB_dpB)
    pAB = primal(pAB_dpAB)
    conjA = primal(conjA_dconjA)
    conjB = primal(conjB_dconjB)
    α = primal(α_dα)
    β = primal(β_dβ)
    ba = primal.(ba_dba)
    C_cache = copy(C)
    TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)
    function contract_pb(::NoRData)
        scale!(C, C_cache, One())
        if Tα == Zero && Tβ == Zero
            scale!(dC, zero(TC))
            return ntuple(i -> NoRData(), 11 + length(ba))
        end
        ipAB = invperm(linearize(pAB))
        pdC = (
            TupleTools.getindices(ipAB, trivtuple(numout(pA))),
            TupleTools.getindices(ipAB, numout(pA) .+ trivtuple(numin(pB))),
        )
        ipA = (invperm(linearize(pA)), ())
        ipB = (invperm(linearize(pB)), ())
        conjΔC = conjA
        conjB′ = conjA ? conjB : !conjB
        dA = tensorcontract!(
            dA,
            dC, pdC, conjΔC,
            B, reverse(pB), conjB′,
            ipA,
            conjA ? α : conj(α), One(), ba...
        )
        conjΔC = conjB
        conjA′ = conjB ? conjA : !conjA
        dB = tensorcontract!(
            dB,
            A, reverse(pA), conjA′,
            dC, pdC, conjΔC,
            ipB,
            conjB ? α : conj(α), One(), ba...
        )
        dα = if _needs_tangent(Tα)
            C_αβ = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
            # TODO: consider using `inner`
            Mooncake._rdata(
                tensorscalar(
                    tensorcontract(
                        C_αβ, ((), trivtuple(numind(pAB))), true,
                        dC, (trivtuple(numind(pAB)), ()), false,
                        ((), ()), One(), ba...
                    )
                )
            )
        else
            NoRData()
        end
        dβ = if _needs_tangent(Tβ)
            # TODO: consider using `inner`
            Mooncake._rdata(
                tensorscalar(
                    tensorcontract(
                        C, ((), trivtuple(numind(pAB))), true,
                        dC, (trivtuple(numind(pAB)), ()), false,
                        ((), ()), One(), ba...
                    )
                )
            )
        else
            NoRData()
        end
        if β === Zero()
            scale!(dC, β)
        else
            scale!(dC, conj(β))
        end
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, contract_pb
end

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(tensoradd!), AbstractArray, AbstractArray, Index2Tuple, Bool, Number, Number, Vararg{Any}}
function Mooncake.rrule!!(
        ::CoDual{typeof(tensoradd!)},
        C_dC::CoDual{<:AbstractArray{TC}},
        A_dA::CoDual{<:AbstractArray{TA}},
        pA_dpA::CoDual{<:Index2Tuple},
        conjA_dconjA::CoDual{Bool},
        α_dα::CoDual{Tα},
        β_dβ::CoDual{Tβ},
        ba_dba::CoDual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    C, dC = arrayify(C_dC)
    A, dA = arrayify(A_dA)
    pA = primal(pA_dpA)
    conjA = primal(conjA_dconjA)
    α = primal(α_dα)
    β = primal(β_dβ)
    ba = primal.(ba_dba)
    C_cache = copy(C)
    TensorOperations.tensoradd!(C, A, pA, conjA, α, β, ba...)
    function add_pb(::NoRData)
        scale!(C, C_cache, One())
        ipA = invperm(linearize(pA))
        dA = tensoradd!(dA, dC, (ipA, ()), conjA, conjA ? α : conj(α), One(), ba...)
        dα = if _needs_tangent(Tα)
            tensorscalar(
                tensorcontract(
                    A, ((), linearize(pA)), !conjA,
                    dC, (trivtuple(numind(pA)), ()), false,
                    ((), ()), One(), ba...
                )
            )
        else
            Mooncake.NoRData()
        end
        dβ = if _needs_tangent(Tβ)
            tensorscalar(
                tensorcontract(
                    C, ((), trivtuple(numind(pA))), true,
                    dC, (trivtuple(numind(pA)), ()), false,
                    ((), ()), One(), ba...
                )
            )
        else
            Mooncake.NoRData()
        end
        if β === Zero()
            scale!(dC, β)
        else
            scale!(dC, conj(β))
        end
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, add_pb
end

Mooncake.@is_primitive DefaultCtx ReverseMode Tuple{typeof(tensortrace!), AbstractArray, AbstractArray, Index2Tuple, Index2Tuple, Bool, Number, Number, Vararg{Any}}
function Mooncake.rrule!!(
        ::CoDual{typeof(tensortrace!)},
        C_dC::CoDual{<:AbstractArray{TC}},
        A_dA::CoDual{<:AbstractArray{TA}},
        p_dp::CoDual{<:Index2Tuple},
        q_dq::CoDual{<:Index2Tuple},
        conjA_dconjA::CoDual{Bool},
        α_dα::CoDual{Tα},
        β_dβ::CoDual{Tβ},
        ba_dba::CoDual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    C, dC = arrayify(C_dC)
    A, dA = arrayify(A_dA)
    p = primal(p_dp)
    q = primal(q_dq)
    conjA = primal(conjA_dconjA)
    α = primal(α_dα)
    β = primal(β_dβ)
    ba = primal.(ba_dba)
    C_cache = copy(C)
    TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, ba...)
    function trace_pb(::NoRData)
        scale!(C, C_cache, One())
        ip = invperm((linearize(p)..., q[1]..., q[2]...))
        Es = map(q[1], q[2]) do i1, i2
            one(
                TensorOperations.tensoralloc_add(
                    TensorOperations.scalartype(A), A, ((i1,), (i2,)), conjA
                )
            )
        end
        E = _kron(Es, ba)
        dA = tensorproduct!(
            dA, dC, (trivtuple(numind(p)), ()), conjA,
            E, ((), trivtuple(numind(q))), conjA,
            (ip, ()),
            conjA ? α : conj(α), One(), ba...
        )
        C_αβ = tensortrace(A, p, q, false, One(), ba...)
        dα = if _needs_tangent(Tα)
            Mooncake._rdata(
                tensorscalar(
                    tensorcontract(
                        C_αβ, ((), trivtuple(numind(p))),
                        !conjA,
                        dC, (trivtuple(numind(p)), ()), false,
                        ((), ()), One(), ba...
                    )
                )
            )
        else
            NoRData()
        end
        dβ = if _needs_tangent(Tβ)
            Mooncake._rdata(
                tensorscalar(
                    tensorcontract(
                        C, ((), trivtuple(numind(p))), true,
                        dC, (trivtuple(numind(p)), ()), false,
                        ((), ()), One(), ba...
                    )
                )
            )
        else
            NoRData()
        end
        if β === Zero()
            scale!(dC, β)
        else
            scale!(dC, conj(β))
        end
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, trace_pb
end

end
