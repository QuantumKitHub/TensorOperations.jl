module TensorOperationsEnzymeExt

using TensorOperations
using TensorOperations: AbstractBackend, DefaultAllocator, CUDAAllocator, ManualAllocator
using VectorInterface
using TupleTools
using Enzyme, ChainRulesCore
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules

@inline EnzymeRules.inactive(::typeof(TensorOperations.tensorfree!), ::Any) = true
Enzyme.@import_rrule(typeof(TensorOperations.tensoralloc), Any, Any, Any, Any)

@inline EnzymeRules.inactive_type(v::Type{<:AbstractBackend}) = true
@inline EnzymeRules.inactive_type(v::Type{DefaultAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{<:CUDAAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{ManualAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{<:Index2Tuple}) = true

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorOperations.tensorcontract!)},
        ::Type{RT},
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        pA_dpA::Const{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        B_dB::Annotation{<:AbstractArray{TB}},
        pB_dpB::Const{<:Index2Tuple},
        conjB_dconjB::Const{Bool},
        pAB_dpAB::Const{<:Index2Tuple},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TB <: Number, TC <: Number}
    # form caches if needed
    cache_A = !isa(A_dA, Const) && EnzymeRules.overwritten(config)[3] ? copy(A_dA.val) : nothing
    cache_B = !isa(B_dB, Const) && EnzymeRules.overwritten(config)[6] ? copy(B_dB.val) : nothing
    cache_C = copy(C_dC.val) # do we need to do this, if we don't need the primal?
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    TensorOperations.tensorcontract!(C_dC.val, A_dA.val, pA_dpA.val, conjA_dconjA.val, B_dB.val, pB_dpB.val, conjB_dconjB.val, pAB_dpAB.val, α_dα.val, β_dβ.val, ba...)
    primal = if EnzymeRules.needs_primal(config)
        C_dC.val
    else
        nothing
    end
    shadow = EnzymeRules.needs_shadow(config) ? C_dC.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_B, cache_C))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorOperations.tensorcontract!)},
        ::Type{RT},
        cache,
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        pA_dpA::Const{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        B_dB::Annotation{<:AbstractArray{TB}},
        pB_dpB::Const{<:Index2Tuple},
        conjB_dconjB::Const{Bool},
        pAB_dpAB::Const{<:Index2Tuple},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TB <: Number, TC <: Number}
    cache_A, cache_B, cache_C = cache
    Aval = something(cache_A, A_dA.val)
    Bval = something(cache_B, B_dB.val)
    Cval = cache_C
    dC = C_dC.dval
    dA = A_dA.dval
    dB = B_dB.dval
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    dC, dA, dB, dα, dβ = TensorOperations.tensorcontract_pullback!(dC, dA, dB, Cval, Aval, pA_dpA.val, conjA_dconjA.val, Bval, pB_dpB.val, conjB_dconjB.val, pAB_dpAB.val, α, β, ba...)
    return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, dα, dβ, map(ba_ -> nothing, ba)...
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Annotation{typeof(tensoradd!)},
        ::Type{RT},
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        pA_dpA::Const{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    # form caches if needed
    cache_A = EnzymeRules.overwritten(config)[3] ? copy(A_dA.val) : nothing
    cache_C = !iszero(β_dβ.val) ? copy(C_dC.val) : nothing
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    conjA = conjA_dconjA.val
    TensorOperations.tensoradd!(C_dC.val, A_dA.val, pA_dpA.val, conjA, α, β, ba...)
    primal = if EnzymeRules.needs_primal(config)
        C_dC.val
    else
        nothing
    end
    shadow = EnzymeRules.needs_shadow(config) ? C_dC.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_C))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Annotation{typeof(tensoradd!)},
        ::Type{RT},
        cache,
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        pA_dpA::Const{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    cache_A, cache_C = cache
    Aval = something(cache_A, A_dA.val)
    Cval = cache_C
    pA = pA_dpA.val
    conjA = conjA_dconjA.val
    α = α_dα.val
    β = β_dβ.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    dC = C_dC.dval
    dA = A_dA.dval
    dC, dA, dα, dβ = TensorOperations.tensoradd_pullback!(dC, dA, Cval, Aval, pA, conjA, α, β, ba...)
    return nothing, nothing, nothing, nothing, dα, dβ, map(ba_ -> nothing, ba)...
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Annotation{typeof(tensortrace!)},
        ::Type{RT},
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        p_dp::Const{<:Index2Tuple},
        q_dq::Const{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    # form caches if needed
    cache_A = EnzymeRules.overwritten(config)[3] ? copy(A_dA.val) : nothing
    cache_C = !iszero(β_dβ.val) ? copy(C_dC.val) : nothing
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    conjA = conjA_dconjA.val
    TensorOperations.tensortrace!(C_dC.val, A_dA.val, p_dp.val, q_dq.val, conjA, α, β, ba...)
    primal = if EnzymeRules.needs_primal(config)
        C_dC.val
    else
        nothing
    end
    shadow = EnzymeRules.needs_shadow(config) ? C_dC.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (cache_A, cache_C))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Annotation{typeof(tensortrace!)},
        ::Type{RT},
        cache,
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        p_dp::Const{<:Index2Tuple},
        q_dq::Const{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    cache_A, cache_C = cache
    Aval = something(cache_A, A_dA.val)
    Cval = cache_C
    p = p_dp.val
    q = q_dq.val
    conjA = conjA_dconjA.val
    α = α_dα.val
    β = β_dβ.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    dC = C_dC.dval
    dA = A_dA.dval
    dC, dA, dα, dβ = TensorOperations.tensortrace_pullback!(dC, dA, Cval, Aval, p, q, conjA, α, β, ba...)
    return nothing, nothing, nothing, nothing, nothing, dα, dβ, map(ba_ -> nothing, ba)...
end

end
