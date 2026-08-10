module TensorOperationsGPUArraysExt

using GPUArrays
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# BufferAllocator with AbstractGPUArray storage
#-------------------------------------------------------------------------------------------

const GPUBufferAllocator = TO.BufferAllocator{<:AbstractGPUArray}

# `GPUArrays.derive` is the documented backend hook for producing an array of a different
# type and size backed by the same data, and is what `reshape` and contiguous `view`s go
# through. Sharing the buffer's refcounted storage keeps it alive for as long as the
# temporary is, and going through `derive` rather than a backend's own constructor or
# `unsafe_wrap` keeps this insensitive to how a backend represents the offset internally.
# The offset `derive` takes is counted in elements, which the padding that `tensoralloc`
# applies guarantees the byte offset to be a whole number of.
function TO.unsafe_buffer_wrap(
        ::Type{A}, buffer::GPUBufferAllocator, start, structure
    ) where {A <: AbstractGPUArray}
    T = eltype(A)
    return GPUArrays.derive(T, buffer.buffer, TO._asdims(structure), Int(start) ÷ sizeof(T))
end

end
