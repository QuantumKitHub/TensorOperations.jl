module TensorOperationsEnzymeExt

using TensorOperations
using TensorOperations: numind, numin, numout, promote_contract
using TensorOperations: AbstractBackend, DefaultAllocator, CUDAAllocator, ManualAllocator
using VectorInterface
using TupleTools
using Enzyme
using Enzyme.EnzymeCore
using Enzyme.EnzymeCore: EnzymeRules

trivtuple(N) = ntuple(identity, N)

# To avoid computing rrules for α and β when these aren't needed, we want to have a
# type-stable quick bail-out
_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{<:Number}) = true
_needs_tangent(::Type{<:Integer}) = false
_needs_tangent(::Type{<:Union{One, Zero}}) = false

_kron(Es::NTuple{1}, ba) = Es[1]
function _kron(Es::NTuple{N, Any}, ba) where {N}
    E1 = Es[1]
    E2 = _kron(Base.tail(Es), ba)
    p2 = ((), trivtuple(2 * N - 2))
    p = ((1, (2 .+ trivtuple(N - 1))...), (2, ((N + 1) .+ trivtuple(N - 1))...))
    return tensorproduct(p, E1, ((1, 2), ()), false, E2, p2, false, One(), ba...)
end

@inline EnzymeRules.inactive_type(v::Type{<:AbstractBackend}) = true
@inline EnzymeRules.inactive_type(v::Type{DefaultAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{CUDAAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{ManualAllocator}) = true
@inline EnzymeRules.inactive_type(v::Type{Index2Tuple}) = true

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
    cache_C = copy(C_dC.val)
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    primal = if EnzymeRules.needs_primal(config)
        TensorOperations.tensorcontract!(C_dC.val, A_dA.val, pA_dpA.val, conjA_dconjA.val, B_dB.val, pB_dpB.val, conjB_dconjB.val, pAB_dpAB.val, α_dα.val, β_dβ.val, ba...)
        C_dC.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        C_dC.dval
    else
        nothing
    end
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
    ΔC, ΔA, ΔB, dα, dβ = tensorcontract_pb!(dC, Cval, dA, Aval, dB, Bval, α, β, pA_dpA.val, pB_dpB.val, pAB_dpAB.val, conjA_dconjA.val, conjB_dconjB.val, ba...)
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
    cache_A = !isa(A_dA, Const) && EnzymeRules.overwritten(config)[3] ? copy(A_dA.val) : nothing
    cache_C = copy(C_dC.val)
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    conjA = conjA_dconjA.val
    primal = if EnzymeRules.needs_primal(config)
        TensorOperations.tensoradd!(C_dC.val, A_dA.val, pA_dpA.val, conjA, α, β, ba...)
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        C_dC.dval
    else
        nothing
    end
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
    ipA = invperm(linearize(pA))
    conjA = conjA_dconjA.val
    α = α_dα.val
    β = β_dβ.val
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    tensoradd!(A_dA.dval, C_dC.dval, (ipA, ()), conjA, conjA ? α : conj(α), One(), ba...)
    dα = if !isa(α_dα, Const) && _needs_tangent(Tα)
        tensorscalar(
            tensorcontract(
                Aval, ((), linearize(pA)), !conjA,
                C_dC.dval, (trivtuple(numind(pA)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    dβ = if !isa(β_dβ, Const) &&  _needs_tangent(Tβ)
        tensorscalar(
            tensorcontract(
                Cval, ((), trivtuple(numind(pA))), true,
                C_dC.dval, (trivtuple(numind(pA)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    if β === Zero()
        scale!(C_dC.dval, β)
    else
        scale!(C_dC.dval, conj(β))
    end
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
    cache_A = !isa(A_dA, Const) && EnzymeRules.overwritten(config)[3] ? copy(A_dA.val) : nothing
    cache_C = copy(C_dC.val)
    ba = map(ba_ -> getfield(ba_, :val), ba_dba)
    α = α_dα.val
    β = β_dβ.val
    conjA = conjA_dconjA.val
    primal = if EnzymeRules.needs_primal(config)
        TensorOperations.tensortrace!(C_dC.val, A_dA.val, p_dp.val, q_dq.val, conjA, α, β, ba...)
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        C_dC.dval
    else
        nothing
    end
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
    ip = invperm((linearize(p)..., q[1]..., q[2]...))
    Es = map(q[1], q[2]) do i1, i2
        one(
            TensorOperations.tensoralloc_add(
                TensorOperations.scalartype(Aval), Aval, ((i1,), (i2,)), conjA
            )
        )
    end
    E = _kron(Es, ba)
    dA = tensorproduct!(
        A_dA.dval, C_dC.dval, (trivtuple(numind(p)), ()), conjA,
        E, ((), trivtuple(numind(q))), conjA,
        (ip, ()),
        conjA ? α : conj(α), One(), ba...
    )
    C_αβ = tensortrace(Aval, p, q, false, One(), ba...)
    dα = if !isa(α_dα, Const) && _needs_tangent(Tα)
        tensorscalar(
            tensorcontract(
                C_αβ, ((), trivtuple(numind(p))),
                !conjA,
                C_dC.dval, (trivtuple(numind(p)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    dβ = if !isa(β_dβ, Const) && _needs_tangent(Tβ)
        tensorscalar(
            tensorcontract(
                Cval, ((), trivtuple(numind(p))), true,
                C_dC.dval, (trivtuple(numind(p)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    if β === Zero()
        scale!(C_dC.dval, β)
    else
        scale!(C_dC.dval, conj(β))
    end
    return nothing, nothing, nothing, nothing, nothing, dα, dβ, map(ba_ -> nothing, ba)...
end

end
