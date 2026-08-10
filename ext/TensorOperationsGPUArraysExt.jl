module TensorOperationsGPUArraysExt

using GPUArrays
using TensorOperations
using TensorOperations: TensorOperations as TO

#-------------------------------------------------------------------------------------------
# BufferAllocator with AbstractGPUArray storage
#-------------------------------------------------------------------------------------------

const GPUBufferAllocator = TO.BufferAllocator{<:AbstractGPUArray}

# `GPUArrays.derive` is the backend hook that `reshape` and contiguous `view`s go through: it
# shares the buffer's refcounted storage, which keeps it alive for as long as the temporary,
# and is insensitive to how a backend represents the offset internally
function TO.unsafe_buffer_wrap(
        ::Type{A}, buffer::GPUBufferAllocator, start, structure
    ) where {A <: AbstractGPUArray}
    T = eltype(A)
    return GPUArrays.derive(T, buffer.buffer, TO._asdims(structure), Int(start) ÷ sizeof(T))
end

end
