module TensorOperationscuTENSORExt

using TensorOperations
using TensorOperations: TensorOperations as TO
using TensorOperations: cuTENSORBackend, CUDAAllocator, DefaultAllocator
using TensorOperations: isstrided

using cuTENSOR
using cuTENSOR: OP_IDENTITY, OP_CONJ, OP_ADD
using cuTENSOR: is_unary, is_binary
using cuTENSOR: handle, stream
using cuTENSOR: cutensorWorksizePreference_t, cutensorAlgo_t, cutensorOperationDescriptor_t,
    cutensorOperator_t, cutensorJitMode_t, cutensorPlanPreference_t,
    cutensorComputeDescriptorEnum
using cuTENSOR: WORKSPACE_DEFAULT, ALGO_DEFAULT, JIT_MODE_NONE
using cuTENSOR: cutensorCreatePlanPreference, cutensorPlan, CuTensorPlan,
    CuTensorDescriptor, ModeType

using cuTENSOR: elementwise_binary_execute!, permute!, contract!, reduce!

# reduce
using cuTENSOR: reduction_compute_types, cutensorCreateReduction

using cuTENSOR: CUDA
using CUDA: CuArray, StridedCuArray, DenseCuArray, AnyCuArray
# this might be dependency-piracy, but removes a dependency from the main package
using CUDA.Adapt: adapt

using Strided
using TupleTools: TupleTools as TT

# Disallowed paradigm from Julia 1.11.1 onwards:
# const StridedViewsCUDAExt = @static if isdefined(Base, :get_extension)
#     Base.get_extension(Strided.StridedViews, :StridedViewsCUDAExt)
# else
#     Strided.StridedViews.StridedViewsCUDAExt
# end
# isnothing(StridedViewsCUDAExt) && error("StridedViewsCUDAExt not found")

# Literal copy of the StridedViewsCUDAExt module
const CuStridedView{T, N, A <: CuArray{T}} = StridedView{T, N, A}

#-------------------------------------------------------------------------------------------
# @cutensor macro
#-------------------------------------------------------------------------------------------
function TensorOperations._cutensor(src, ex...)
    # TODO: there is no check for doubled tensor kwargs
    return Expr(
        :macrocall, GlobalRef(TensorOperations, Symbol("@tensor")),
        src,
        Expr(:(=), :backend, Expr(:call, GlobalRef(TensorOperations, :cuTENSORBackend))),
        Expr(:(=), :allocator, Expr(:call, GlobalRef(TensorOperations, :CUDAAllocator))),
        ex...
    )
end

#-------------------------------------------------------------------------------------------
# Backend selection and passing
#-------------------------------------------------------------------------------------------
# A Base wrapper over `CuArray` will first pass via the `select_backend` methods for
# `AbstractArray` and be converted into a `StridedView` if it satisfies `isstrided`. Hence,
# we only need to capture `CuStridedView` here.
function TO.select_backend(::typeof(TO.tensoradd!), ::CuStridedView, ::CuStridedView)
    return cuTENSORBackend()
end
function TO.select_backend(::typeof(TO.tensortrace!), ::CuStridedView, ::CuStridedView)
    return cuTENSORBackend()
end
function TO.select_backend(
        ::typeof(TO.tensorcontract!), ::CuStridedView, ::CuStridedView, ::CuStridedView
    )
    return cuTENSORBackend()
end

# TODO: with `CUDA.HostMemory` and `unsafe_wrap(CuArray, ::Array)` we could in principle
# support mixed argument lists with some `CuStridedView` and some `HostStridedView`, but
# I am not sure if we want to go that way.

# Make sure that if the `cuTensorBackend` is specified, arrays are converted to CuArrays
function TO.tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::cuTENSORBackend, allocator = CUDAAllocator()
    )
    C_cuda, isview = _custrided(C, allocator)
    A_cuda, = _custrided(A, allocator)
    tensoradd!(C_cuda, A_cuda, pA, conjA, α, β, backend, allocator)
    isview || copy!(C, C_cuda.parent)
    return C
end
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::cuTENSORBackend, allocator = CUDAAllocator()
    )
    C_cuda, isview = _custrided(C, allocator)
    A_cuda, = _custrided(A, allocator)
    B_cuda, = _custrided(B, allocator)
    tensorcontract!(
        C_cuda, A_cuda, pA, conjA, B_cuda, pB, conjB, pAB, α, β, backend, allocator
    )
    isview || copy!(C, C_cuda.parent)
    return C
end
function TO.tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::cuTENSORBackend, allocator = CUDAAllocator()
    )
    C_cuda, isview = _custrided(C, allocator)
    A_cuda, = _custrided(A, allocator)
    tensortrace!(C_cuda, A_cuda, p, q, conjA, α, β, backend, allocator)
    isview || copy!(C, C_cuda.parent)
    return C
end

_custrided(A::AbstractArray, ::DefaultAllocator) = _custrided(A, CUDAAllocator())
function _custrided(
        A::AbstractArray, allocator::CUDAAllocator{Mout, Min, Mtemp}
    ) where {Mout, Min, Mtemp}
    if isstrided(A)
        return _custrided(StridedView(A), allocator)
    else
        return StridedView(CuArray{eltype(A), ndims(A), Mtemp}(A)), false
    end
end
function _custrided(
        A::StridedView, allocator::CUDAAllocator{Mout, Min, Mtemp}
    ) where {Mout, Min, Mtemp}
    P = A.parent
    if P isa CuArray
        return A, true
    elseif P isa Array && Min === CUDA.HostMemory
        P_cuda = unsafe_wrap(CuArray, P)
        return StridedView(P_cuda, A.size, A.strides, A.offset, A.op), true
    else
        P_cuda = CuArray{eltype(P), ndims(P), Mtemp}(P)
        return StridedView(P_cuda, A.size, A.strides, A.offset, A.op), false
    end
end

#-------------------------------------------------------------------------------------------
# Allocator
#-------------------------------------------------------------------------------------------
function TO.CUDAAllocator()
    Mout = CUDA.UnifiedMemory
    Min = CUDA.default_memory
    Mtemp = CUDA.default_memory
    return CUDAAllocator{Mout, Min, Mtemp}()
end

function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::CUDAAllocator
    )
    ttype = CuArray{TC, TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val, allocator::CUDAAllocator
    )
    ttype = CuArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# Overwrite tensoradd_type
function TO.tensoradd_type(TC, A::CuArray, pA::Index2Tuple, conjA::Bool)
    return CuArray{TC, sum(length.(pA))}
end

# NOTE: the general implementation in the `DefaultAllocator` case works just fine, without
# selecting an explicit memory model
function TO.tensoralloc(
        ::Type{CuArray{T, N}}, structure,
        ::Val{istemp}, allocator::CUDAAllocator{Mout, Min, Mtemp}
    ) where {T, N, istemp, Mout, Min, Mtemp}
    M = istemp ? Mtemp : Mout
    return CuArray{T, N, M}(undef, structure)
end

function TO.tensorfree!(C::CuArray, ::CUDAAllocator)
    CUDA.unsafe_free!(C)
    return nothing
end

#-------------------------------------------------------------------------------------------
# Implementation
#-------------------------------------------------------------------------------------------
function TO.tensorscalar(C::CuStridedView)
    return ndims(C) == 0 ? CUDA.@allowscalar(C[]) : throw(DimensionMismatch())
end

function tensorop(A::CuStridedView, conjA::Bool = false)
    return (eltype(A) <: Real || !xor(conjA, A.op === conj)) ? OP_IDENTITY : OP_CONJ
end

function TO.tensoradd!(
        C::CuStridedView,
        A::CuStridedView, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::cuTENSORBackend, allocator
    )
    # convert arguments
    Ainds, Cinds = collect.(TO.add_labels(pA))
    opA = tensorop(A, conjA)

    # dispatch to cuTENSOR
    return if iszero(β)
        permute!(α, A, Ainds, opA, C, Cinds)
    else
        elementwise_binary_execute!(
            α,
            A, Ainds, opA,
            β,
            C, Cinds, OP_IDENTITY,
            C, Cinds,
            OP_ADD
        )
    end
end

function TO.tensorcontract!(
        C::CuStridedView,
        A::CuStridedView, pA::Index2Tuple, conjA::Bool,
        B::CuStridedView, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::cuTENSORBackend, allocator
    )

    # convert arguments
    Ainds, Binds, Cinds = collect.(TO.contract_labels(pA, pB, pAB))
    opA = tensorop(A, conjA)
    opB = tensorop(B, conjB)

    # dispatch to cuTENSOR
    return contract!(
        α,
        A, Ainds, opA,
        B, Binds, opB,
        β,
        C, Cinds, OP_IDENTITY,
        OP_IDENTITY
    )
end

function TO.tensortrace!(
        C::CuStridedView,
        A::CuStridedView, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::cuTENSORBackend, allocator
    )
    # convert arguments
    Ainds, Cinds = collect.(TO.trace_labels(p, q))
    opA = tensorop(A, conjA)

    # map to reduction operation
    plan = plan_trace(A, Ainds, opA, C, Cinds, OP_IDENTITY, OP_ADD)
    return reduce!(plan, α, A, β, C)
end

function cuTENSOR.CuTensorDescriptor(
        a::CuStridedView;
        size = size(a), strides = strides(a), eltype = eltype(a)
    )
    sz = collect(Int64, size)
    st = collect(Int64, strides)
    alignment = UInt32(find_alignment(a))
    return cuTENSOR.CuTensorDescriptor(sz, st, eltype, alignment)
end

const MAX_ALIGNMENT = UInt(256) # This is the largest alignment of CUDA memory
"find the alignment of the first element of the view"
find_alignment(A::CuStridedView) = gcd(MAX_ALIGNMENT, convert(UInt, pointer(A)))

# trace!
# ------
# not actually part of cuTENSOR, just a special case of reduce
function plan_trace(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        jit::cutensorJitMode_t = JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t = WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t = ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing} = nothing
    )
    !is_unary(opA) && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC) && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opReduce) && throw(ArgumentError("opReduce must be a binary op!"))

    # TODO: check if this can be avoided, available in caller
    # TODO: cuTENSOR will allocate sizes and strides anyways, could use that here
    p, q = TO.trace_indices(tuple(Ainds...), tuple(Cinds...))
    qsorted = TT.sort(q[2])
    # add strides of cindA2 to strides of cindA1 -> selects diagonal
    stA = strides(A)
    for (i, j) in zip(q...)
        stA = Base.setindex(stA, stA[i] + stA[j], i)
    end
    szA = TT.deleteat(size(A), qsorted)
    stA′ = TT.deleteat(stA, qsorted)

    descA = CuTensorDescriptor(A; size = szA, strides = stA′)
    descC = CuTensorDescriptor(C)

    modeA = collect(Cint, deleteat!(Ainds, qsorted))
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        reduction_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateReduction(
        handle(),
        desc,
        descA, modeA, opA,
        descC, modeC, opC,
        descC, modeC, opReduce,
        actual_compute_type
    )

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref = workspace)
    cuTENSOR.cutensorDestroyOperationDescriptor(desc[])
    cuTENSOR.cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

end
