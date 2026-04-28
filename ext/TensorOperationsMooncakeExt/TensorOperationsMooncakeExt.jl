module TensorOperationsMooncakeExt

using TensorOperations
# Mooncake imports ChainRulesCore as CRC to avoid name conflicts
# here we import it ourselves to ensure the rules from the ChainRulesCore
# extension are in fact loaded
using Mooncake, Mooncake.CRC
using TensorOperations: AbstractBackend, DefaultAllocator, CUDAAllocator, ManualAllocator
using TensorOperations: tensoralloc, tensoradd!, tensorcontract!, tensortrace!
using Mooncake: ReverseMode, DefaultCtx, Dual, CoDual, NoRData, arrayify, @zero_derivative, primal, tangent
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

Mooncake.@is_primitive DefaultCtx Tuple{typeof(tensorcontract!), AbstractArray, AbstractArray, Index2Tuple, Bool, AbstractArray, Index2Tuple, Bool, Index2Tuple, Number, Number, Vararg{Any}}
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
        dα = isnothing(Δα) ? NoRData() : Δα
        dβ = isnothing(Δβ) ? NoRData() : Δβ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, contract_pb
end

function Mooncake.frule!!(
        ::Dual{typeof(tensorcontract!)},
        C_dC::Dual{<:AbstractArray{TC}},
        A_dA::Dual{<:AbstractArray{TA}},
        pA_dpA::Dual{<:Index2Tuple},
        conjA_dconjA::Dual{Bool},
        B_dB::Dual{<:AbstractArray{TB}},
        pB_dpB::Dual{<:Index2Tuple},
        conjB_dconjB::Dual{Bool},
        pAB_dpAB::Dual{<:Index2Tuple},
        α_dα::Dual{Tα},
        β_dβ::Dual{Tβ},
        ba_dba::Dual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TB <: Number, TC <: Number}
    C, dC = arrayify(C_dC)
    A, dA = arrayify(A_dA)
    B, dB = arrayify(B_dB)
    pA = primal(pA_dpA)
    pB = primal(pB_dpB)
    pAB = primal(pAB_dpAB)
    conjA = primal(conjA_dconjA)
    conjB = primal(conjB_dconjB)
    α, dα = Mooncake.extract(α_dα)
    β, dβ = Mooncake.extract(β_dβ)
    ba = primal.(ba_dba)

    # ΔC′ = ΔC*β + C*Δβ + A*B*Δα + ΔA*B*α + A*ΔB*α
    scale!(dC, β)
    if !isa(dβ, Mooncake.NoTangent)
        @. dC += dβ * C
    end
    if !isa(dα, Mooncake.NoTangent)
        tensorcontract!(dC, A, pA, conjA, B, pB, conjB, pAB, dα, One(), ba...)
    end
    tensorcontract!(dC, dA, pA, conjA, B, pB, conjB, pAB, α, One(), ba...)
    tensorcontract!(dC, A, pA, conjA, dB, pB, conjB, pAB, α, One(), ba...)
    TensorOperations.tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)
    return C_dC
end

Mooncake.@is_primitive DefaultCtx Tuple{typeof(tensoradd!), AbstractArray, AbstractArray, Index2Tuple, Bool, Number, Number, Vararg{Any}}
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
        dC, dA, Δα, Δβ = TensorOperations.tensoradd_pullback!(dC, dA, C, A, pA, conjA, α, β, ba...)
        dα = isnothing(Δα) ? NoRData() : Δα
        dβ = isnothing(Δβ) ? NoRData() : Δβ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, add_pb
end

function Mooncake.frule!!(
        ::Dual{typeof(tensoradd!)},
        C_dC::Dual{<:AbstractArray{TC}},
        A_dA::Dual{<:AbstractArray{TA}},
        pA_dpA::Dual{<:Index2Tuple},
        conjA_dconjA::Dual{Bool},
        α_dα::Dual{Tα},
        β_dβ::Dual{Tβ},
        ba_dba::Dual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    C, dC = arrayify(C_dC)
    A, dA = arrayify(A_dA)
    pA = primal(pA_dpA)
    conjA = primal(conjA_dconjA)
    α = primal(α_dα)
    dα = tangent(α_dα)
    β = primal(β_dβ)
    dβ = tangent(β_dβ)
    ba = primal.(ba_dba)
    # D = α * A + β *C

    # dD = dα * A + α * dA + β dC + dβ * C

    # dC′ = β dC + dβ * C
    scale!(dC, β)
    if !isa(dβ, Mooncake.NoTangent)
        @. dC += dβ * C
    end
    TensorOperations.tensoradd!(dC, dA, pA, conjA, α, One(), ba...)
    if !isa(dα, Mooncake.NoTangent)
        TensorOperations.tensoradd!(dC, A, pA, conjA, dα, One(), ba...)
    end
    TensorOperations.tensoradd!(C, A, pA, conjA, α, β, ba...)
    return C_dC
end

Mooncake.@is_primitive DefaultCtx Tuple{typeof(tensortrace!), AbstractArray, AbstractArray, Index2Tuple, Index2Tuple, Bool, Number, Number, Vararg{Any}}
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
        dα = isnothing(Δα) ? NoRData() : Δα
        dβ = isnothing(Δβ) ? NoRData() : Δβ
        return NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), NoRData(), dα, dβ, map(ba_ -> NoRData(), ba)...
    end
    return C_dC, trace_pb
end

function Mooncake.frule!!(
        ::Dual{typeof(tensortrace!)},
        C_dC::Dual{<:AbstractArray{TC}},
        A_dA::Dual{<:AbstractArray{TA}},
        p_dp::Dual{<:Index2Tuple},
        q_dq::Dual{<:Index2Tuple},
        conjA_dconjA::Dual{Bool},
        α_dα::Dual{Tα},
        β_dβ::Dual{Tβ},
        ba_dba::Dual...,
    ) where {Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    C, dC = arrayify(C_dC)
    A, dA = arrayify(A_dA)
    p = primal(p_dp)
    q = primal(q_dq)
    conjA = primal(conjA_dconjA)
    α = primal(α_dα)
    dα = tangent(α_dα)
    β = primal(β_dβ)
    dβ = tangent(β_dβ)
    ba = primal.(ba_dba)
    # dD = dα * tr(A) + α * tr(dA) + dβ * C + β * dC
    # dC1 = dβ * C + β * dC
    scale!(dC, β)
    if !isa(dβ, Mooncake.NoTangent)
        @. dC += dβ * C
    end
    if !isa(dα, Mooncake.NoTangent)
        TensorOperations.tensortrace!(dC, A, p, q, conjA, dα, One(), ba...)
    end
    TensorOperations.tensortrace!(dC, dA, p, q, conjA, α, One(), ba...)
    TensorOperations.tensortrace!(C, A, p, q, conjA, α, β, ba...)
    return C_dC
end

end
