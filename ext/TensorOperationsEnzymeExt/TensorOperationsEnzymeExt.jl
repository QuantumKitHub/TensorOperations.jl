module TensorOperationsEnzymeExt

using TensorOperations
using TensorOperations: AbstractBackend, DefaultAllocator, CUDAAllocator, ManualAllocator
using VectorInterface
using TupleTools
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules

@inline EnzymeRules.inactive(::typeof(TensorOperations.tensorfree!), ::Any) = true
@inline EnzymeRules.inactive_type(v::Type{<:AbstractBackend}) = true
@inline EnzymeRules.inactive_type(v::Type{DefaultAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{<:CUDAAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{ManualAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{<:Index2Tuple}) = true
@inline EnzymeRules.inactive_type(v::Type{<:IndexTuple}) = true

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorOperations.tensoralloc)},
        ::Type{RT},
        ttype::Const,
        structure::Const,
        istemp::Const{Bool},
        allocator::Const
    ) where {RT}
    primal = EnzymeRules.needs_primal(config) ? TensorOperations.tensoralloc(ttype.val, structure.val, Val(false), allocator.val) : nothing
    shadow = EnzymeRules.needs_shadow(config) ? TensorOperations.tensoralloc(ttype.val, structure.val, Val(false), allocator.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(TensorOperations.tensoralloc)},
        ::Type{RT},
        cache,
        ttype::Const,
        structure::Const,
        istemp::Const{Bool},
        allocator::Const,
    ) where {RT}
    return nothing, nothing, nothing, nothing
end

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
    cache_A = EnzymeRules.overwritten(config)[3] ? copy(A_dA.val) : nothing
    cache_B = EnzymeRules.overwritten(config)[6] ? copy(B_dB.val) : nothing
    cache_C = !isa(β_dβ, Const) ? copy(C_dC.val) : C_dC.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    TensorOperations.tensorcontract!(C_dC.val, A_dA.val, pA_dpA.val, conjA_dconjA.val, B_dB.val, pB_dpB.val, conjB_dconjB.val, pAB_dpAB.val, α_dα.val, β_dβ.val, ba...)
    primal = EnzymeRules.needs_primal(config) ? C_dC.val : nothing
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
    # good way to check that we don't use it accidentally when we should not be needing it?
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    pA, pB, pAB, conjA, conjB = getfield.((pA_dpA, pB_dpB, pAB_dpAB, conjA_dconjA, conjB_dconjB), :val)

    if !isa(A_dA, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        ΔA = A_dA.dval
        TensorOperations.tensorcontract_pullback_dA!(ΔA, ΔC, Cval, Aval, pA, conjA, Bval, pB, conjB, pAB, α, ba...)
    end
    if !isa(B_dB, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        ΔB = B_dB.dval
        TensorOperations.tensorcontract_pullback_dB!(ΔB, ΔC, Cval, Aval, pA, conjA, Bval, pB, conjB, pAB, α, ba...)
    end
    Δα = if !isa(α_dα, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        TensorOperations.tensorcontract_pullback_dα(ΔC, Cval, Aval, pA, conjA, Bval, pB, conjB, pAB, α, ba...)
    elseif !isa(α_dα, Const)
        zero(α_dα.val)
    else
        nothing
    end
    Δβ = if !isa(β_dβ, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        TensorOperations.pullback_dβ(ΔC, Cval, β)
    elseif !isa(β_dβ, Const)
        zero(β_dβ.val)
    else
        nothing
    end
    !isa(C_dC, Const) && TensorOperations.pullback_dC!(C_dC.dval, β)
    return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, Δα, Δβ, map(ba_ -> nothing, ba)...
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(TensorOperations.tensorcontract!)},
        ::Type{RT},
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        pA_dpA::Annotation{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        B_dB::Annotation{<:AbstractArray{TB}},
        pB_dpB::Annotation{<:Index2Tuple},
        conjB_dconjB::Const{Bool},
        pAB_dpAB::Annotation{<:Index2Tuple},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TB <: Number, TC <: Number}
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    pA, pB, pAB, conjA, conjB = getfield.((pA_dpA, pB_dpB, pAB_dpAB, conjA_dconjA, conjB_dconjB), :val)

    if !isa(C_dC, Const)
        scale!(C_dC.dval, β)
        if !isa(β_dβ, Const)
            @. C_dC.dval += β_dβ.dval * C_dC.val
        end
        if !isa(α_dα, Const)
            tensorcontract!(C_dC.dval, A_dA.val, pA, conjA, B_dB.val, pB, conjB, pAB, α_dα.dval, One(), ba...)
        end
        if !isa(A_dA, Const)
            tensorcontract!(C_dC.dval, A_dA.dval, pA, conjA, B_dB.val, pB, conjB, pAB, α, One(), ba...)
        end
        if !isa(B_dB, Const)
            tensorcontract!(C_dC.dval, A_dA.val, pA, conjA, B_dB.dval, pB, conjB, pAB, α, One(), ba...):Zero()
        end
    end
    TensorOperations.tensorcontract!(C_dC.val, A_dA.val, pA, conjA, B_dB.val, pB, conjB, pAB, α, β, ba...)
    return C_dC
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
    cache_C = !iszero(β_dβ.val) ? copy(C_dC.val) : C_dC.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    conjA = conjA_dconjA.val
    TensorOperations.tensoradd!(C_dC.val, A_dA.val, pA_dpA.val, conjA, α, β, ba...)
    primal = EnzymeRules.needs_primal(config) ? C_dC.val : nothing
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

    if !isa(A_dA, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        ΔA = A_dA.dval
        TensorOperations.tensoradd_pullback_dA!(ΔA, ΔC, Cval, Aval, pA, conjA, α, ba...)
    end
    Δα = if !isa(α_dα, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        TensorOperations.tensoradd_pullback_dα(ΔC, Cval, Aval, pA, conjA, α, ba...)
    elseif !isa(α_dα, Const)
        zero(α_dα.val)
    else
        nothing
    end
    Δβ = if !isa(β_dβ, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        TensorOperations.pullback_dβ(ΔC, Cval, β)
    elseif !isa(β_dβ, Const)
        zero(β_dβ.val)
    else
        nothing
    end
    !isa(C_dC, Const) && TensorOperations.pullback_dC!(C_dC.dval, β)
    return nothing, nothing, nothing, nothing, Δα, Δβ, map(ba_ -> nothing, ba)...
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1},
        ::Annotation{typeof(tensoradd!)},
        ::Type{RT},
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        pA_dpA::Annotation{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    pA = pA_dpA.val
    conjA = conjA_dconjA.val
    α = α_dα.val
    β = β_dβ.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)

    # D = α * A + β *C

    # dD = dα * A + α * dA + β dC + dβ * C

    # dC′ = β dC + dβ * C
    if !isa(C_dC, Const)
        scale!(C_dC.dval, β)
        if !isa(β_dβ, Const)
            @. C_dC.dval += β_dβ.dval * C_dC.val
        end
        if !isa(A_dA, Const)
            TensorOperations.tensoradd!(C_dC.dval, A_dA.dval, pA, conjA, α, One(), ba...)
        end
        if !isa(α_dα, Const)
            TensorOperations.tensoradd!(C_dC.dval, A_dA.val, pA, conjA, α_dα.dval, One(), ba...)
        end
    end
    TensorOperations.tensoradd!(C_dC.val, A_dA.val, pA, conjA, α, β, ba...)
    return C_dC
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
    cache_C = !isa(β_dβ, Const) ? copy(C_dC.val) : nothing
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    conjA = conjA_dconjA.val
    TensorOperations.tensortrace!(C_dC.val, A_dA.val, p_dp.val, q_dq.val, conjA, α, β, ba...)
    primal = EnzymeRules.needs_primal(config) ? C_dC.val : nothing
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
    Cval = something(cache_C, C_dC.val)
    p = p_dp.val
    q = q_dq.val
    conjA = conjA_dconjA.val
    α = α_dα.val
    β = β_dβ.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)

    if !isa(A_dA, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        ΔA = A_dA.dval
        TensorOperations.tensortrace_pullback_dA!(ΔA, ΔC, Cval, Aval, p, q, conjA, α, ba...)
    end
    Δα = if !isa(α_dα, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        TensorOperations.tensortrace_pullback_dα(ΔC, Cval, Aval, p, q, conjA, α, ba...)
    elseif !isa(α_dα, Const)
        zero(α_dα.val)
    else
        nothing
    end
    Δβ = if !isa(β_dβ, Const) && !isa(C_dC, Const)
        ΔC = C_dC.dval
        TensorOperations.pullback_dβ(ΔC, Cval, β)
    elseif !isa(β_dβ, Const)
        zero(β_dβ.val)
    else
        nothing
    end
    !isa(C_dC, Const) && TensorOperations.pullback_dC!(C_dC.dval, β)
    return nothing, nothing, nothing, nothing, nothing, Δα, Δβ, map(ba_ -> nothing, ba)...
end

function EnzymeRules.forward(
        config::EnzymeRules.RevConfigWidth{1},
        ::Annotation{typeof(tensortrace!)},
        ::Type{RT},
        C_dC::Annotation{<:AbstractArray{TC}},
        A_dA::Annotation{<:AbstractArray{TA}},
        p_dp::Annotation{<:Index2Tuple},
        q_dq::Annotation{<:Index2Tuple},
        conjA_dconjA::Const{Bool},
        α_dα::Annotation{Tα},
        β_dβ::Annotation{Tβ},
        ba_dba::Const...,
    ) where {RT, Tα <: Number, Tβ <: Number, TA <: Number, TC <: Number}
    p = p_dp.val
    q = q_dq.val
    conjA = conjA_dconjA.val
    α = α_dα.val
    β = β_dβ.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)

    # dD = dα * tr(A) + α * tr(dA) + dβ * C + β * dC
    # dC1 = dβ * C + β * dC
    if !isa(C_dC, Const)
        scale!(C_dC.dval, β)
        if !isa(β_dβ, Const)
            @. C_dC, dval += β_dβ.dval * C_dC.val
        end
        if !isa(α_dα, Const)
            TensorOperations.tensortrace!(C_dC.dval, A_dA.val, p, q, conjA, α_dα.dval, One(), ba...)
        end
        if !isa(A_dA, Const)
            TensorOperations.tensortrace!(C_dC.dval, A_dA.dval, p, q, conjA, α, One(), ba...)
        end
    end
    # D = α * tr(A) + β * C
    TensorOperations.tensortrace!(C_dC.val, A_dA.val, p, q, conjA, α, β, ba...)
    return C_dC
end

end
