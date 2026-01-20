module TensorOperationsMooncakeExt

using TensorOperations
# Mooncake imports ChainRulesCore as CRC to avoid name conflicts
# here we import it ourselves to ensure the rules from the ChainRulesCore
# extension are in fact loaded
using Mooncake, Mooncake.CRC
using TensorOperations: AbstractBackend, DefaultAllocator, CUDAAllocator, ManualAllocator
using TensorOperations: tensoralloc, tensoradd!, tensorcontract!, tensortrace!
using Mooncake: ReverseMode, DefaultCtx, CoDual, NoRData, arrayify, @zero_derivative, primal, tangent
using VectorInterface, TupleTools

Mooncake.tangent_type(::Type{Index2Tuple}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:AbstractBackend}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{DefaultAllocator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{CUDAAllocator}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{ManualAllocator}) = Mooncake.NoTangent

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
        dC, dA, dB, Δα, Δβ = TensorOperations.tensorcontract_pullback!(dC, dA, dB, C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)
        dα = isnothing(Δα) ? NoRData() : Mooncake._rdata(Δα)
        dβ = isnothing(Δβ) ? NoRData() : Mooncake._rdata(Δβ)
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
    pA    = primal(pA_dpA)
    conjA = primal(conjA_dconjA)
    α = primal(α_dα)
    β = primal(β_dβ)
    ba = primal.(ba_dba)
    C_cache = copy(C)
    TensorOperations.tensoradd!(C, A, pA, conjA, α, β, ba...)
    function add_pb(::NoRData)
        scale!(C, C_cache, One())
        dC, dA, Δα, Δβ = TensorOperations.tensoradd_pullback!(dC, dA, C, A, pA, conjA, α, β, ba...)
        dα = isnothing(Δα) ? NoRData() : Mooncake._rdata(Δα)
        dβ = isnothing(Δβ) ? NoRData() : Mooncake._rdata(Δβ)
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
        dC, dA, Δα, Δβ = TensorOperations.tensortrace_pullback!(dC, dA, C, A, p, q, conjA, α, β, ba...)
        dα = isnothing(Δα) ? NoRData() : Mooncake._rdata(Δα)
        dβ = isnothing(Δβ) ? NoRData() : Mooncake._rdata(Δβ)
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, trace_pb
end

end
