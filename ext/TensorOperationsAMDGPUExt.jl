module TensorOperationsAMDGPUExt

using AMDGPU
using AMDGPU.hipTENSOR
using TensorOperations
using TensorOperations: TensorOperations as TO
using TensorOperations: hipTENSORBackend, AMDAllocator, DefaultAllocator
using TensorOperations: isstrided

using Strided
using TupleTools: TupleTools as TT

hiptensor_working = AMDGPU.hipTENSOR.has_hiptensor()

using AMDGPU.hipTENSOR: OP_IDENTITY, OP_CONJ, OP_ADD
using AMDGPU.hipTENSOR: is_unary, is_binary
using AMDGPU.hipTENSOR: hiptensorWorksizePreference_t, hiptensorAlgo_t, hiptensorOperationDescriptor_t,
    hiptensorOperator_t, hiptensorJitMode_t, hiptensorPlanPreference_t
using AMDGPU.hipTENSOR: WORKSPACE_DEFAULT, ALGO_DEFAULT, JIT_MODE_NONE
using AMDGPU.hipTENSOR: hiptensorCreatePlanPreference, hiptensorPlan, hipTensorPlan,
    hipTensorDescriptor, ModeType

using AMDGPU.hipTENSOR: elementwise_binary_execute!, permute!, contract!, reduce!
using AMDGPU.hipTENSOR: reduction_compute_types, hiptensorCreateReduction
using AMDGPU.Adapt: adapt

#-------------------------------------------------------------------------------------------
# Allocator
#-------------------------------------------------------------------------------------------

TO.tensoradd_type(TC, A::AnyROCArray, pA::Index2Tuple, conjA::Bool) =
    ROCArray{TC, TO.numind(pA)}

function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::TO.AMDAllocator
    )
    ttype = ROCArray{TC, TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val, allocator::TO.AMDAllocator
    )
    ttype = ROCArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: the general implementation in the `DefaultAllocator` case works just fine, without
# selecting an explicit memory model
function TO.tensoralloc(
        ::Type{<:ROCArray{T, N}}, structure,
        ::Val{istemp}, allocator::TO.AMDAllocator
    ) where {T, N, istemp}
    return ROCArray{T, N}(undef, structure)
end

function TO.tensorfree!(C::ROCArray, ::TO.AMDAllocator)
    AMDGPU.unsafe_free!(C)
    return nothing
end

#-------------------------------------------------------------------------------------------
# BufferAllocator with ROCArray storage
#-------------------------------------------------------------------------------------------

const ROCBufferAllocator{B} = TO.BufferAllocator{ROCArray{UInt8, 1, B}}

# Note: separate binding from the alias above, as a type alias cannot be added to the parent module from an extension
function TO.AMDBufferAllocator(;
        sizehint::Integer = 0, buftype = AMDGPU.Mem.HIPBuffer
    )
    return TO.BufferAllocator{ROCArray{UInt8, 1, buftype}}(; sizehint)
end

# HIP allocations are 256-byte aligned; matching that keeps rocBLAS kernel selection identical, at ≤255 bytes of padding
TO.buffer_alignment(::ROCBufferAllocator) = 256

# mirror the `AMDAllocator` behavior: results and temporaries are `ROCArray`s, even if the inputs are regular host arrays
function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::ROCBufferAllocator
    )
    ttype = ROCArray{TC, TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val, allocator::ROCBufferAllocator
    )
    ttype = ROCArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: for tensors backed by the buffer this only releases the reference that `unsafe_buffer_wrap` retained
function TO.tensorfree!(C::ROCArray, ::ROCBufferAllocator)
    AMDGPU.unsafe_free!(C)
    return nothing
end

function Base.resize!(buffer::ROCBufferAllocator{B}, n::Integer) where {B}
    isempty(buffer) || error("Cannot resize a buffer that still contains elements")
    n = TO._buffersz(n)
    if n != length(buffer)
        AMDGPU.unsafe_free!(buffer.buffer) # free before allocating new one to reduce memory pressure
        buffer.buffer = ROCArray{UInt8, 1, B}(undef, n)
    end
    return buffer
end

# Literal copy of the StridedViewsAMDGPUExt module
const ROCStridedView{T, N, A <: ROCArray{T}} = StridedView{T, N, A}

#-------------------------------------------------------------------------------------------
# @hiptensor macro
#-------------------------------------------------------------------------------------------
function TensorOperations._hiptensor(src, ex...)
    # TODO: there is no check for doubled tensor kwargs
    return Expr(
        :macrocall, GlobalRef(TensorOperations, Symbol("@tensor")),
        src,
        Expr(:(=), :backend, Expr(:call, GlobalRef(TensorOperations, :hipTENSORBackend))),
        Expr(:(=), :allocator, Expr(:call, GlobalRef(TensorOperations, :AMDAllocator))),
        ex...
    )
end

#-------------------------------------------------------------------------------------------
# Backend selection and passing
#-------------------------------------------------------------------------------------------
# A Base wrapper over `ROCArray` will first pass via the `select_backend` methods for
# `AbstractArray` and be converted into a `StridedView` if it satisfies `isstrided`. Hence,
# we only need to capture `ROCStridedView` here. TODO handle the case where hipTensor isn't
# functional.
function TO.select_backend(::typeof(TO.tensoradd!), ::ROCStridedView, ::ROCStridedView)
    return hipTENSORBackend()
end
function TO.select_backend(::typeof(TO.tensortrace!), ::ROCStridedView, ::ROCStridedView)
    return hipTENSORBackend()
end
function TO.select_backend(
        ::typeof(TO.tensorcontract!), ::ROCStridedView, ::ROCStridedView, ::ROCStridedView
    )
    return hipTENSORBackend()
end

# Make sure that if the `hipTensorBackend` is specified, arrays are converted to ROCArrays
function TO.tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::hipTENSORBackend, allocator = AMDAllocator()
    )
    C_hip, isview = _hipstrided(C, allocator)
    A_hip, = _hipstrided(A, allocator)
    tensoradd!(C_hip, A_hip, pA, conjA, α, β, backend, allocator)
    isview || copy!(C, C_hip.parent)
    return C
end
function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::hipTENSORBackend, allocator = AMDAllocator()
    )
    C_hip, isview = _hipstrided(C, allocator)
    A_hip, = _hipstrided(A, allocator)
    B_hip, = _hipstrided(B, allocator)
    tensorcontract!(
        C_hip, A_hip, pA, conjA, B_hip, pB, conjB, pAB, α, β, backend, allocator
    )
    isview || copy!(C, C_hip.parent)
    return C
end
function TO.tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::hipTENSORBackend, allocator = AMDAllocator()
    )
    C_hip, isview = _hipstrided(C, allocator)
    A_hip, = _hipstrided(A, allocator)
    tensortrace!(C_hip, A_hip, p, q, conjA, α, β, backend, allocator)
    isview || copy!(C, C_hip.parent)
    return C
end

_hipstrided(A::AbstractArray, ::DefaultAllocator) = _hipstrided(A, AMDAllocator())
# input tensors are marshalled to the device in the same way as for the default allocator:
# the buffer is reserved for the temporaries that are created by the tensor operations
_hipstrided(A::AbstractArray, ::TO.BufferAllocator) = _hipstrided(A, AMDAllocator())
function _hipstrided(A::AbstractArray, allocator::AMDAllocator)
    if isstrided(A)
        return _hipstrided(StridedView(A), allocator)
    else
        return StridedView(ROCArray{eltype(A), ndims(A)}(A)), false
    end
end
function _hipstrided(
        A::StridedView, allocator::AMDAllocator
    )
    P = A.parent
    if P isa ROCArray
        return A, true
    else
        P_hip = ROCArray{eltype(P), ndims(P)}(P)
        return StridedView(P_hip, A.size, A.strides, A.offset, A.op), false
    end
end

#-------------------------------------------------------------------------------------------
# Implementation
#-------------------------------------------------------------------------------------------
function TO.tensorscalar(C::ROCStridedView)
    return ndims(C) == 0 ? AMDGPU.@allowscalar(C[]) : throw(DimensionMismatch())
end

function tensorop(A::ROCStridedView, conjA::Bool = false)
    return (eltype(A) <: Real || !xor(conjA, A.op === conj)) ? OP_IDENTITY : OP_CONJ
end

isdense(A::ROCStridedView) = strides(A) == Base.size_to_strides(1, size(A)...)

# hipTENSOR ignores `OP_CONJ` in contractions, so conjugated operands are materialized
function materialize_conj(A::ROCStridedView, conjA::Bool)
    tensorop(A, conjA) == OP_IDENTITY && return A
    A′ = StridedView(ROCArray{eltype(A)}(undef, size(A)))
    A′ .= conjA ? conj.(A) : A
    return A′
end

function TO.tensoradd!(
        C::ROCStridedView,
        A::ROCStridedView, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::hipTENSORBackend, allocator
    )
    # hipTENSOR does not validate sizes itself
    TO.argcheck_tensoradd(C, A, TO.linearize(pA))
    TO.dimcheck_tensoradd(C, A, TO.linearize(pA))

    # hipTENSOR's permutations and elementwise operations ignore the layout of the output,
    # so go through a dense temporary instead
    if !isdense(C)
        C′ = StridedView(ROCArray{eltype(C)}(undef, size(C)))
        iszero(β) || (C′ .= C)
        TO.tensoradd!(C′, A, pA, conjA, α, β, backend, allocator)
        C .= C′
        AMDGPU.unsafe_free!(C′.parent)
        return C
    end

    # convert arguments
    Ainds, Cinds = collect.(TO.add_labels(pA))
    opA = tensorop(A, conjA)

    # dispatch to hipTENSOR
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
        C::ROCStridedView,
        A::ROCStridedView, pA::Index2Tuple, conjA::Bool,
        B::ROCStridedView, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::hipTENSORBackend, allocator
    )
    # hipTENSOR does not validate sizes itself
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    # convert arguments
    Ainds, Binds, Cinds = collect.(TO.contract_labels(pA, pB, pAB))
    A′ = materialize_conj(A, conjA)
    B′ = materialize_conj(B, conjB)

    # hipTENSOR reads C even if β is zero, so NaNs in uninitialized memory would propagate
    iszero(β) && (C .= zero(eltype(C)))

    # dispatch to hipTENSOR
    contract!(
        α,
        A′, Ainds, OP_IDENTITY,
        B′, Binds, OP_IDENTITY,
        β,
        C, Cinds, OP_IDENTITY,
        OP_IDENTITY
    )
    A′ === A || AMDGPU.unsafe_free!(A′.parent)
    B′ === B || AMDGPU.unsafe_free!(B′.parent)
    return C
end

function TO.tensortrace!(
        C::ROCStridedView,
        A::ROCStridedView, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::hipTENSORBackend, allocator
    )
    # hipTENSOR does not validate sizes itself
    TO.argcheck_tensortrace(C, A, TO.linearize(p), q)
    TO.dimcheck_tensortrace(C, A, TO.linearize(p), q)

    # convert arguments
    Ainds, Cinds = collect.(TO.trace_labels(p, q))
    opA = tensorop(A, conjA)

    # map to reduction operation
    plan = plan_trace(A, Ainds, opA, C, Cinds, OP_IDENTITY, OP_ADD)
    reduce!(plan, α, A, β, C)
    AMDGPU.unsafe_free!(plan) # the plan-taking `reduce!` does not free the plan
    return C
end

function hipTENSOR.hipTensorDescriptor(
        a::ROCStridedView;
        size = size(a), strides = strides(a), eltype = eltype(a)
    )
    sz = collect(Int64, size)
    st = collect(Int64, strides)
    alignment = UInt32(find_alignment(a))
    return hipTENSOR.hipTensorDescriptor(sz, st, eltype, alignment)
end

const MAX_ALIGNMENT = UInt(1) # This is the largest alignment of AMD memory
"find the alignment of the first element of the view"
find_alignment(A::ROCStridedView) = gcd(MAX_ALIGNMENT, convert(UInt, pointer(A)))

# trace!
# ------
# not actually part of hipTENSOR, just a special case of reduce
function plan_trace(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::hiptensorOperator_t,
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::hiptensorOperator_t,
        opReduce::hiptensorOperator_t;
        jit::hiptensorJitMode_t = JIT_MODE_NONE,
        workspace::hiptensorWorksizePreference_t = WORKSPACE_DEFAULT,
        algo::hiptensorAlgo_t = ALGO_DEFAULT,
        compute_type::Union{DataType, Nothing} = nothing
    )
    !is_unary(opA) && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC) && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opReduce) && throw(ArgumentError("opReduce must be a binary op!"))

    # TODO: check if this can be avoided, available in caller
    # TODO: hipTENSOR will allocate sizes and strides anyways, could use that here
    p, q = TO.trace_indices(tuple(Ainds...), tuple(Cinds...))
    qsorted = TT.sort(q[2])
    # add strides of cindA2 to strides of cindA1 -> selects diagonal
    stA = strides(A)
    for (i, j) in zip(q...)
        stA = Base.setindex(stA, stA[i] + stA[j], i)
    end
    szA = TT.deleteat(size(A), qsorted)
    stA′ = TT.deleteat(stA, qsorted)

    # hipTENSOR needs A's modes in the same order as C's, followed by the traced ones
    modesA = deleteat!(Ainds, qsorted)
    order = vcat(Cinds, filter(∉(Cinds), modesA))
    perm = map(m -> findfirst(==(m), modesA), order)

    descA = hipTensorDescriptor(A; size = szA[perm], strides = stA′[perm])
    descC = hipTensorDescriptor(C)

    modeA = collect(Cint, order)
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        reduction_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{hiptensorOperationDescriptor_t}()
    hiptensorCreateReduction(
        hipTENSOR.handle(),
        desc,
        descA, modeA, opA,
        descC, modeC, opC,
        descC, modeC, opReduce,
        actual_compute_type
    )

    plan_pref = Ref{hiptensorPlanPreference_t}()
    hiptensorCreatePlanPreference(hipTENSOR.handle(), plan_pref, algo, jit)

    # the plan takes ownership of `desc` and `plan_pref` and destroys them when it is freed
    return hipTensorPlan(desc[], plan_pref[], actual_compute_type, Any[descA, descC, modeA, modeC]; workspacePref = workspace)
end

end
