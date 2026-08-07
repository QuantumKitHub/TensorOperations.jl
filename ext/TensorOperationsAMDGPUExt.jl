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

end
