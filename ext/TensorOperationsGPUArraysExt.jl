module TensorOperationsGPUArraysExt

using GPUArrays
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# BufferAllocator with AbstractGPUArray storage
#-------------------------------------------------------------------------------------------

const GPUBufferAllocator = TO.BufferAllocator{<:AbstractGPUArray}

# A GPU buffer backs exactly the arrays its own storage produces, which `buffer_similartype`
# answers for every backend at once. A derived array can only be offset by a whole number of
# elements, so additionally restrict to element types for which the padded offset is
# expressible that way.
function TO.buffer_arraytype(
        ::Type{A}, buffer::GPUBufferAllocator
    ) where {A <: AbstractArray}
    TO.buffer_iselementaddressable(eltype(A), buffer) || return nothing
    return TO.buffer_similartype(A, buffer)
end

# `GPUArrays.derive` is the documented backend hook for producing an array of a different
# type and size backed by the same data, and is what `reshape` and contiguous `view`s go
# through. Sharing the buffer's refcounted storage keeps it alive for as long as the
# temporary is, and going through `derive` rather than a backend's own constructor or
# `unsafe_wrap` keeps this insensitive to how a backend represents the offset internally.
function TO.unsafe_buffer_wrap(
        ::Type{A}, buffer::GPUBufferAllocator, start, structure
    ) where {A <: AbstractGPUArray}
    T = eltype(A)
    return GPUArrays.derive(T, buffer.buffer, TO._asdims(structure), Int(start) ÷ sizeof(T))
end

end
