module TensorOperationsChainRulesCoreExt

using TensorOperations
using TensorOperations: numind, numin, numout, promote_contract, _needs_tangent
using TensorOperations: pullback_dC, pullback_dβ,
    tensoradd_pullback_dA, tensoradd_pullback_dα,
    tensorcontract_pullback_dA, tensorcontract_pullback_dB, tensorcontract_pullback_dα,
    tensortrace_pullback_dA, tensortrace_pullback_dα

using TensorOperations: DefaultBackend, DefaultAllocator, _kron
using ChainRulesCore
using TupleTools
using VectorInterface
using TupleTools: invperm
using LinearAlgebra

@non_differentiable TensorOperations.tensorstructure(args...)
@non_differentiable TensorOperations.tensoradd_structure(args...)
@non_differentiable TensorOperations.tensoradd_type(args...)
@non_differentiable TensorOperations.tensoralloc_add(args...)
@non_differentiable TensorOperations.tensorcontract_structure(args...)
@non_differentiable TensorOperations.tensorcontract_type(args...)
@non_differentiable TensorOperations.tensoralloc_contract(args...)
@non_differentiable TensorOperations.promote_contract(args...)
@non_differentiable TensorOperations.promote_add(args...)

# Cannot free intermediate tensors when using AD
# Thus we change the forward passes: `istemp=false` and `tensorfree!` is a no-op
function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensorfree!), allocator = DefaultAllocator()
    )
    tensorfree!_pullback(Δargs...) = (NoTangent(), NoTangent())
    return nothing, tensorfree!_pullback
end
function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensoralloc), ttype, structure,
        istemp, allocator = DefaultAllocator()
    )
    output = TensorOperations.tensoralloc(ttype, structure, Val(false), allocator)
    function tensoralloc_pullback(Δargs...)
        return (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return output, tensoralloc_pullback
end

# this function more or less boils down to `fill!(similar(x), y)` but does so in a single
# call to allow higher-order derivatives
function similar_and_fill(x, y)
    x′ = TensorOperations.tensoralloc(typeof(x), TensorOperations.tensorstructure(x))
    return fill!(x′, y)
end
function ChainRulesCore.rrule(::typeof(similar_and_fill), x, y)
    similar_and_fill_pullback(Δx) = NoTangent(), ZeroTangent(), tensorscalar(unthunk(Δx))
    return similar_and_fill(x, y), similar_and_fill_pullback
end
function ChainRulesCore.rrule(::typeof(tensorscalar), C)
    tensorscalar_pullback(Δc) = NoTangent(), similar_and_fill(C, unthunk(Δc))
    return tensorscalar(C), tensorscalar_pullback
end

# The current `rrule` design makes sure that the implementation for custom types does
# not need to support the backend or allocator arguments
function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensoradd!),
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ba...
    )
    return _rrule_tensoradd!(C, A, pA, conjA, α, β, ba)
end
function _rrule_tensoradd!(C, A, pA, conjA, α, β, ba)
    C′ = tensoradd!(copy(C), A, pA, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function tensoradd_pullback(ΔC′)
        ΔC = unthunk(ΔC′)

        dC = β === Zero() ? ZeroTangent() : @thunk projectC(pullback_dC(ΔC, β))
        dA = @thunk projectA(tensoradd_pullback_dA(ΔC, C, A, pA, conjA, α, ba...))
        dα = if _needs_tangent(α)
            @thunk projectα(tensoradd_pullback_dα(ΔC, C, A, pA, conjA, α, ba...))
        else
            ZeroTangent()
        end
        dβ = if _needs_tangent(β)
            @thunk projectβ(pullback_dβ(ΔC, C, β))
        else
            ZeroTangent()
        end
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, tensoradd_pullback
end

function ChainRulesCore.rrule(
        ::typeof(TensorOperations.tensorcontract!),
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ba...
    )
    return _rrule_tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba)
end
function _rrule_tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ba)
    C′ = tensorcontract!(copy(C), A, pA, conjA, B, pB, conjB, pAB, α, β, ba...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function tensorcontract_pullback(ΔC′)
        ΔC = unthunk(ΔC′)

        dC = β === Zero() ? ZeroTangent() : @thunk projectC(pullback_dC(ΔC, β))
        dA = @thunk projectA(tensorcontract_pullback_dA(ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...))
        dB = @thunk projectB(tensorcontract_pullback_dB(ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...))
        dα = if _needs_tangent(α)
            @thunk projectα(tensorcontract_pullback_dα(ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...))
        else
            ZeroTangent()
        end
        dβ = if _needs_tangent(β)
            @thunk projectβ(pullback_dβ(ΔC, C, β))
        else
            ZeroTangent()
        end
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC,
            dA, NoTangent(), NoTangent(),
            dB, NoTangent(), NoTangent(),
            NoTangent(),
            dα, dβ, dba...
    end

    return C′, tensorcontract_pullback
end

function ChainRulesCore.rrule(
        ::typeof(tensortrace!), C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        ba...
    )
    return _rrule_tensortrace!(C, A, p, q, conjA, α, β, ba)
end
function _rrule_tensortrace!(C, A, p, q, conjA, α, β, ba)
    C′ = tensortrace!(copy(C), A, p, q, conjA, α, β, ba...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function tensortrace_pullback(ΔC′)
        ΔC = unthunk(ΔC′)

        dC = β === Zero() ? ZeroTangent() : @thunk projectC(pullback_dC(ΔC, β))
        dA = @thunk projectA(tensortrace_pullback_dA(ΔC, C, A, p, q, conjA, α, ba...))
        dα = if _needs_tangent(α)
            @thunk projectα(tensortrace_pullback_dα(ΔC, C, A, p, q, conjA, α, ba...))
        else
            ZeroTangent()
        end
        dβ = if _needs_tangent(β)
            @thunk projectβ(pullback_dβ(ΔC, C, β))
        else
            ZeroTangent()
        end
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, tensortrace_pullback
end

# NCON functions
@non_differentiable TensorOperations.ncontree(args...)
@non_differentiable TensorOperations.nconoutput(args...)
@non_differentiable TensorOperations.check_nconstyle(args...)
@non_differentiable TensorOperations.indexordertree(args...)

end
