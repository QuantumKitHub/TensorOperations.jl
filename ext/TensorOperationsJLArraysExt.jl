module TensorOperationsJLArraysExt

using JLArrays
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# Allocator
#-------------------------------------------------------------------------------------------

TO.tensoradd_type(TC, A::JLArray, pA::Index2Tuple, conjA::Bool) =
    JLArray{TC, TO.numind(pA)}

#-------------------------------------------------------------------------------------------
# BufferAllocator with JLArray storage
#-------------------------------------------------------------------------------------------

const JLBuffer = TO.BufferAllocator{JLArray{UInt8, 1}}

# Note: separate binding from the alias above, as a type alias cannot be added to the parent module from an extension
TO.JLBufferAllocator(; sizehint::Integer = 0) =
    TO.BufferAllocator{JLArray{UInt8, 1}}(; sizehint)

# Derived arrays are offset by a whole number of elements, so `T`s whose size does not divide
# the alignment cannot be buffer-backed
function _iselementaddressable(::Type{T}, buffer::JLBuffer) where {T}
    sz = sizeof(T)
    alignment = max(Base.datatype_alignment(T), TO.buffer_alignment(buffer))
    return !iszero(sz) && iszero(alignment % sz)
end

# JLArray buffers can only back `JLArray`s; the generic implementation already takes care of
# the converse, i.e. that host buffers can never back `JLArray`s
function TO.buffer_arraytype(::Type{<:JLArray{T, N}}, buffer::JLBuffer) where {T, N}
    return _iselementaddressable(T, buffer) ? JLArray{T, N} : nothing
end
TO.buffer_arraytype(::Type{<:Array}, ::JLBuffer) = nothing

# `GPUArrays.derive` is the documented backend hook for producing an array of a different type
# and size backed by the same data, as `reshape` and contiguous `view`s do, so the buffer
# outlives the temporary. Going through it rather than the `JLArray` constructor keeps this
# insensitive to whether the offset is stored per element or in bytes.
function TO.unsafe_buffer_wrap(
        ::Type{JLArray{T, N}}, buffer::JLBuffer, start, structure
    ) where {T, N}
    return JLArrays.GPUArrays.derive(
        T, buffer.buffer, _asdims(structure), Int(start) ÷ sizeof(T)
    )
end

# `structure` is a shape for arrays, but a bare length is accepted for vectors
_asdims(structure::Base.Dims) = structure
_asdims(n::Integer) = (Int(n),)

# mirror the GPU allocator behavior: results and temporaries are `JLArray`s, even if the inputs are regular host arrays
function TO.tensoralloc_add(
        TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        istemp::Val, allocator::JLBuffer
    )
    ttype = JLArray{TC, TO.numind(pA)}
    structure = TO.tensoradd_structure(A, pA, conjA)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

function TO.tensoralloc_contract(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        istemp::Val, allocator::JLBuffer
    )
    ttype = JLArray{TC, TO.numind(pAB)}
    structure = TO.tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return TO.tensoralloc(ttype, structure, istemp, allocator)::ttype
end

# NOTE: for tensors backed by the buffer this only releases the reference that `unsafe_buffer_wrap` retained
function TO.tensorfree!(C::JLArray, ::JLBuffer)
    JLArrays.unsafe_free!(C)
    return nothing
end

function Base.resize!(buffer::JLBuffer, n::Integer)
    isempty(buffer) || error("Cannot resize a buffer that still contains elements")
    n = TO._buffersz(n)
    if n != length(buffer)
        JLArrays.unsafe_free!(buffer.buffer) # free before allocating new one to reduce memory pressure
        buffer.buffer = JLArray{UInt8, 1}(undef, n)
    end
    return buffer
end

end
