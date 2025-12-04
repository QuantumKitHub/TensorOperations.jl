module TensorOperationsChainRulesCoreExt

using TensorOperations
using TensorOperations: numind, numin, numout, promote_contract
using TensorOperations: DefaultBackend, DefaultAllocator
using ChainRulesCore
using TupleTools
using VectorInterface
using TupleTools: invperm
using LinearAlgebra

trivtuple(N) = ntuple(identity, N)

@non_differentiable TensorOperations.tensorstructure(args...)
@non_differentiable TensorOperations.tensoradd_structure(args...)
@non_differentiable TensorOperations.tensoradd_type(args...)
@non_differentiable TensorOperations.tensoralloc_add(args...)
@non_differentiable TensorOperations.tensorcontract_structure(args...)
@non_differentiable TensorOperations.tensorcontract_type(args...)
@non_differentiable TensorOperations.tensoralloc_contract(args...)
@non_differentiable Base.promote_op(args...)

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

function ChainRulesCore.rrule(::typeof(tensorscalar), C)
    projectC = ProjectTo(C) 
    function tensorscalar_pullback(Δc)
        _Δc = unthunk(Δc)
        return NoTangent(), projectC(_Δc)
    end
    return tensorscalar(C), tensorscalar_pullback
end

# The current `rrule` design makes sure that the implementation for custom types does
# not need to support the backend or allocator arguments
# function ChainRulesCore.rrule(::typeof(TensorOperations.tensoradd!),
#                               C,
#                               A, pA::Index2Tuple, conjA::Bool,
#                               α::Number, β::Number,
#                               backend, allocator)
#     val, pb = _rrule_tensoradd!(C, A, pA, conjA, α, β, (backend, allocator))
#     return val, ΔC -> (pb(ΔC)..., NoTangent(), NoTangent())
# end
# function ChainRulesCore.rrule(::typeof(TensorOperations.tensoradd!),
#                               C,
#                               A, pA::Index2Tuple, conjA::Bool,
#                               α::Number, β::Number,
#                               backend)
#     val, pb = _rrule_tensoradd!(C, A, pA, conjA, α, β, (backend,))
#     return val, ΔC -> (pb(ΔC)..., NoTangent())
# end
# function ChainRulesCore.rrule(::typeof(TensorOperations.tensoradd!),
#                               C,
#                               A, pA::Index2Tuple, conjA::Bool,
#                               α::Number, β::Number)
#     return _rrule_tensoradd!(C, A, pA, conjA, α, β, ())
# end
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

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ipA = invperm(linearize(pA))
            _dA = zerovector(A, VectorInterface.promote_add(ΔC, α))
            _dA = tensoradd!(_dA, ΔC, (ipA, ()), conjA, conjA ? α : conj(α), Zero(), ba...)
            projectA(_dA)
        end
        dα = @thunk let
            _dα = tensorscalar(
                tensorcontract(
                    A, ((), linearize(pA)), !conjA,
                    ΔC, (trivtuple(numind(pA)), ()), false,
                    ((), ()), One(), ba...
                )
            )
            projectα(_dα)
        end
        dβ = @thunk let
            # TODO: consider using `inner`
            _dβ = tensorscalar(
                tensorcontract(
                    C, ((), trivtuple(numind(pA))), true,
                    ΔC, (trivtuple(numind(pA)), ()), false,
                    ((), ()), One(), ba...
                )
            )
            projectβ(_dβ)
        end
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

# function ChainRulesCore.rrule(::typeof(TensorOperations.tensorcontract!),
#                               C,
#                               A, pA::Index2Tuple, conjA::Bool,
#                               B, pB::Index2Tuple, conjB::Bool,
#                               pAB::Index2Tuple,
#                               α::Number, β::Number,
#                               backend, allocator)
#     val, pb = _rrule_tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β,
#                                      (backend, allocator))
#     return val, ΔC -> (pb(ΔC)..., NoTangent(), NoTangent())
# end
# function ChainRulesCore.rrule(::typeof(TensorOperations.tensorcontract!),
#                               C,
#                               A, pA::Index2Tuple, conjA::Bool,
#                               B, pB::Index2Tuple, conjB::Bool,
#                               pAB::Index2Tuple,
#                               α::Number, β::Number,
#                               backend)
#     val, pb = _rrule_tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, (backend,))
#     return val, ΔC -> (pb(ΔC)..., NoTangent())
# end
# function ChainRulesCore.rrule(::typeof(TensorOperations.tensorcontract!),
#                               C,
#                               A, pA::Index2Tuple, conjA::Bool,
#                               B, pB::Index2Tuple, conjB::Bool,
#                               pAB::Index2Tuple,
#                               α::Number, β::Number)
#     return _rrule_tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, ())
# end
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

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipAB = invperm(linearize(pAB))
        pΔC = (
            TupleTools.getindices(ipAB, trivtuple(numout(pA))),
            TupleTools.getindices(ipAB, numout(pA) .+ trivtuple(numin(pB))),
        )
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ipA = (invperm(linearize(pA)), ())
            conjΔC = conjA
            conjB′ = conjA ? conjB : !conjB
            _dA = zerovector(A, promote_contract(scalartype(ΔC), scalartype(B), typeof(α)))
            _dA = tensorcontract!(
                _dA,
                ΔC, pΔC, conjΔC,
                B, reverse(pB), conjB′,
                ipA,
                conjA ? α : conj(α), Zero(), ba...
            )
            projectA(_dA)
        end
        dB = @thunk let
            ipB = (invperm(linearize(pB)), ())
            conjΔC = conjB
            conjA′ = conjB ? conjA : !conjA
            _dB = zerovector(B, promote_contract(scalartype(ΔC), scalartype(A), typeof(α)))
            _dB = tensorcontract!(
                _dB,
                A, reverse(pA), conjA′,
                ΔC, pΔC, conjΔC,
                ipB,
                conjB ? α : conj(α), Zero(), ba...
            )
            projectB(_dB)
        end
        dα = @thunk let
            C_αβ = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
            # TODO: consider using `inner`
            _dα = tensorscalar(
                tensorcontract(
                    C_αβ, ((), trivtuple(numind(pAB))), true,
                    ΔC, (trivtuple(numind(pAB)), ()), false,
                    ((), ()), One(), ba...
                )
            )
            projectα(_dα)
        end
        dβ = @thunk let
            # TODO: consider using `inner`
            _dβ = tensorscalar(
                tensorcontract(
                    C, ((), trivtuple(numind(pAB))), true,
                    ΔC, (trivtuple(numind(pAB)), ()), false,
                    ((), ()), One(), ba...
                )
            )
            projectβ(_dβ)
        end
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC,
            dA, NoTangent(), NoTangent(), dB, NoTangent(), NoTangent(),
            NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

# function ChainRulesCore.rrule(::typeof(tensortrace!), C,
#                               A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
#                               α::Number, β::Number,
#                               backend, allocator)
#     val, pb = _rrule_tensortrace!(C, A, p, q, conjA, α, β, (backend, allocator))
#     return val, ΔC -> (pb(ΔC)..., NoTangent(), NoTangent())
# end
# function ChainRulesCore.rrule(::typeof(tensortrace!), C,
#                               A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
#                               α::Number, β::Number,
#                               backend)
#     val, pb = _rrule_tensortrace!(C, A, p, q, conjA, α, β, (backend,))
#     return val, ΔC -> (pb(ΔC)..., NoTangent())
# end
# function ChainRulesCore.rrule(::typeof(tensortrace!), C,
#                               A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
#                               α::Number, β::Number)
#     return _rrule_tensortrace!(C, A, p, q, conjA, α, β, ())
# end
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

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk let
            ip = invperm((linearize(p)..., q[1]..., q[2]...))
            Es = map(q[1], q[2]) do i1, i2
                    one(
                    TensorOperations.tensoralloc_add(
                        scalartype(A), A, ((i1,), (i2,)), conjA
                    )
                )
            end
            E = _kron(Es, ba)
            _dA = zerovector(A, VectorInterface.promote_scale(ΔC, α))
            _dA = tensorproduct!(
                _dA, ΔC, (trivtuple(numind(p)), ()), conjA,
                E, ((), trivtuple(numind(q))), conjA,
                (ip, ()),
                conjA ? α : conj(α), Zero(), ba...
            )
            projectA(_dA)
        end
        dα = @thunk let
            C_αβ = tensortrace(A, p, q, false, One(), ba...)
            _dα = tensorscalar(
                tensorcontract(
                    C_αβ, ((), trivtuple(numind(p))),
                    !conjA,
                    ΔC, (trivtuple(numind(p)), ()), false,
                    ((), ()), One(), ba...
                )
            )
            projectα(_dα)
        end
        dβ = @thunk let
            _dβ = tensorscalar(
                tensorcontract(
                    C, ((), trivtuple(numind(p))), true,
                    ΔC, (trivtuple(numind(p)), ()), false,
                    ((), ()), One(), ba...
                )
            )
            projectβ(_dβ)
        end
        dba = map(_ -> NoTangent(), ba)
        return NoTangent(), dC, dA, NoTangent(), NoTangent(), NoTangent(), dα, dβ, dba...
    end

    return C′, pullback
end

_kron(Es::NTuple{1}, ba) = Es[1]
function _kron(Es::NTuple{N, Any}, ba) where {N}
    E1 = Es[1]
    E2 = _kron(Base.tail(Es), ba)
    p2 = ((), trivtuple(2 * N - 2))
    p = ((1, (2 .+ trivtuple(N - 1))...), (2, ((N + 1) .+ trivtuple(N - 1))...))
    return tensorproduct(p, E1, ((1, 2), ()), false, E2, p2, false, One(), ba...)
end

# NCON functions
@non_differentiable TensorOperations.ncontree(args...)
@non_differentiable TensorOperations.nconoutput(args...)
@non_differentiable TensorOperations.check_nconstyle(args...)
@non_differentiable TensorOperations.indexordertree(args...)

end
