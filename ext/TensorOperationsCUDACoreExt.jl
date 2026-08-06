module TensorOperationsCUDACoreExt

using CUDACore
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# Allocator
#-------------------------------------------------------------------------------------------

TO.tensoradd_type(TC, A::CuArray, pA::Index2Tuple, conjA::Bool) =
    CuArray{TC, TO.numind(pA)}

function TO.CUDAAllocator()
    Mout = CUDACore.UnifiedMemory
    Min = CUDACore.default_memory
    Mtemp = CUDACore.default_memory
    return TO.CUDAAllocator{Mout, Min, Mtemp}()
end

function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::TO.CUDAAllocator
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
        istemp::Val, allocator::TO.CUDAAllocator
    )
    ttype = CuArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: the general implementation in the `DefaultAllocator` case works just fine, without
# selecting an explicit memory model
function TO.tensoralloc(
        ::Type{CuArray{T, N}}, structure,
        ::Val{istemp}, allocator::TO.CUDAAllocator{Mout, Min, Mtemp}
    ) where {T, N, istemp, Mout, Min, Mtemp}
    M = istemp ? Mtemp : Mout
    return CuArray{T, N, M}(undef, structure)
end

function TO.tensorfree!(C::CuArray, ::TO.CUDAAllocator)
    CUDACore.unsafe_free!(C)
    return nothing
end

#-------------------------------------------------------------------------------------------
# BufferAllocator with CuArray storage
#-------------------------------------------------------------------------------------------

const CuBufferAllocator{M} = TO.BufferAllocator{CuArray{UInt8, 1, M}}

# Note: different binding of the same name because type alias cannot be exported from extension
function TO.CUDABufferAllocator(; sizehint::Integer = 0, memory = CUDACore.default_memory)
    return TO.BufferAllocator{CuArray{UInt8, 1, memory}}(; sizehint)
end

# CUDA buffers can only back `CuArray`s; the generic implementation already takes care of
# the converse, i.e. that host buffers can never back `CuArray`s
function TO.buffer_arraytype(::Type{<:CuArray{T, N}}, ::CuBufferAllocator{M}) where {T, N, M}
    return CuArray{T, N, M}
end
TO.buffer_arraytype(::Type{<:Array}, ::CuBufferAllocator) = nothing

# CUDA allocations are 256-byte aligned, and cuTENSOR selects noticeably faster kernels for 256-byte aligned data.
# The padding this costs is at most 255 bytes per temporary, which is negligible in comparison.
TO.buffer_alignment(::CuBufferAllocator) = 256

function TO.unsafe_buffer_wrap(
        ::Type{CuArray{T, N, M}}, buffer::CuBufferAllocator{M}, start, structure
    ) where {T, N, M}
    ptr = convert(CuPtr{T}, pointer(buffer, start))
    return unsafe_wrap(CuArray{T, N, M}, ptr, structure)
end

# mirror the `CUDAAllocator` behavior: results and temporaries are `CuArray`s, even if the inputs are regular host arrays
function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::CuBufferAllocator
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
        istemp::Val, allocator::CuBufferAllocator
    )
    ttype = CuArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: this is a no-op for tensors that are backed by the buffer, as `unsafe_wrap` creates a non-owning reference
function TO.tensorfree!(C::CuArray, ::CuBufferAllocator)
    CUDACore.unsafe_free!(C)
    return nothing
end

function Base.resize!(buffer::CuBufferAllocator{M}, n::Integer) where {M}
    isempty(buffer) || error("Cannot resize a buffer that still contains elements")
    n = TO._buffersz(n)
    if n != length(buffer)
        CUDACore.unsafe_free!(buffer.buffer) # free before allocating new one to reduce memory pressure
        buffer.buffer = CuArray{UInt8, 1, M}(undef, n)
    end
    return buffer
end

end
