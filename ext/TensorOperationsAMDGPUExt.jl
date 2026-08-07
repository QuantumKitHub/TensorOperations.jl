module TensorOperationsAMDGPUExt

using AMDGPU
using TensorOperations
using TensorOperations: TensorOperations as TO

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

# AMD buffers can only back `ROCArray`s; the generic implementation already takes care of
# the converse, i.e. that host buffers can never back `ROCArray`s
function TO.buffer_arraytype(
        ::Type{<:ROCArray{T, N}}, ::ROCBufferAllocator{B}
    ) where {T, N, B}
    return ROCArray{T, N, B}
end
TO.buffer_arraytype(::Type{<:Array}, ::ROCBufferAllocator) = nothing

# HIP allocations are 256-byte aligned; matching that keeps rocBLAS kernel selection identical, at ≤255 bytes of padding
TO.buffer_alignment(::ROCBufferAllocator) = 256

# Share the buffer's refcounted `DataRef` at a byte offset, as `reshape` does: that keeps the buffer alive, and
# avoids the `hipPointerGetAttributes` query that `unsafe_wrap` would do per temporary
function TO.unsafe_buffer_wrap(
        ::Type{ROCArray{T, N, B}}, buffer::ROCBufferAllocator{B}, start, structure
    ) where {T, N, B}
    ref = copy(AMDGPU.GPUArrays.storage(buffer.buffer))
    return ROCArray{T, N}(ref, _asdims(structure); offset = Int(start))
end

# `structure` is a shape for arrays, but a bare length is accepted for vectors
_asdims(structure::Base.Dims) = structure
_asdims(n::Integer) = (Int(n),)

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

end
