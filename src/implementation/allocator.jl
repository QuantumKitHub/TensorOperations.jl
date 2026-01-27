# ------------------------------------------------------------------------------------------
# Allocator backends
# ------------------------------------------------------------------------------------------
"""
    DefaultAllocator()

Default allocator for tensor operations if no explicit allocator is specified. This will
just use the standard constructor for the tensor type, and thus probably uses Julia's
default memory manager.
"""
struct DefaultAllocator end

"""
    CUDAAllocator{Mout,Min,Mtemp}()

Allocator that uses the CUDA memory manager and will thus allocate `CuArray` instances. The
parameters `Min`, `Mout`, `Mtemp` can be any of the CUDA.jl memory types, i.e. 
`CUDA.DeviceMemory`, `CUDA.UnifiedMemory` or `CUDA.HostMemory`.
* `Mout` is used to determine how to deal with output tensors; with `Mout=CUDA.HostMemory`
  or `Mout=CUDA.UnifiedMemory` the CUDA runtime will ensure that the data is also available
  at in the host memory, and can thus be converted back to normal arrays using
  `unsafe_wrap(Array, outputtensor)`. If `Mout=CUDA.DeviceMemory` the data will remain on
  the GPU, untill an explict `Array(outputtensor)` is called.
* `Min` is used to determine how to deal with input tensors; with `Min=CUDA.HostMemory` the
  CUDA runtime will itself take care of transferring the data to the GPU, otherwise it is
  copied explicitly.
* `Mtemp` is used to allocate space for temporary tensors; it defaults to
  `CUDA.default_memory` which is `CUDA.DeviceMemory`. Only if many or huge temporary tensors
  are expected could it be useful to choose `CUDA.UnifiedMemory`.
"""
struct CUDAAllocator{Mout, Min, Mtemp} end

"""
    ManualAllocator()

Allocator that bypasses Julia's memory management for temporary tensors by leveraging `Libc.malloc`
and `Libc.free` directly. This can be useful for reducing the pressure on the garbage collector.
This backend will allocate using `DefaultAllocator` for output tensors that escape the `@tensor`
block, which will thus still be managed using Julia's GC. The other tensors will be backed by
`PtrArray` instances, from `PtrArrays.jl`, thus requiring compatibility with that interface.
"""
struct ManualAllocator end

"""
    BufferAllocator(; sizehint = 0)

Allocator that uses a pre-allocated buffer for storing temporary tensors.
When the buffer is full, the allocator falls back on Julia's default allocation mechanism
to create temporary tensors, but keeps track of how much additional memory is required.
When the buffer is fully reset, the buffer is automatically resized to ensure subsequent
contractions will now fit in the buffer.

!!! warning
    This allocator is **not** thread-safe, and it is the user's responsibility to avoid running
    the same allocator on concurrent jobs.
"""
mutable struct BufferAllocator{Storage}
    buffer::Storage
    offset::UInt
    max_offset::UInt

    function BufferAllocator{Storage}(; sizehint::Integer = 0) where {Storage}
        T = eltype(Storage)
        (isbitstype(T) && sizeof(T) == 1) ||
            throw(ArgumentError("Buffer should have elements that take up a single byte."))
        n = _buffersz(sizehint)
        return new{Storage}(Storage(undef, n), 0, 0)
    end
end

const DefaultStorageType = @static isdefined(Core, :Memory) ? Memory{UInt8} : Vector{UInt8}
BufferAllocator(; kwargs...) = BufferAllocator{DefaultStorageType}(; kwargs...)

# allocate buffers in sizes that are powers of 2
_buffersz(x::Integer) = iszero(x) ? x : Base.nextpow(2, x)

# ------------------------------------------------------------------------------------------
# Generic implementation
# ------------------------------------------------------------------------------------------

# function that mimicks the operations that are applied to the scalars during contraction
tensorop(args...) = +(*(args...), *(args...))

"""
    promote_contract(args...)

Promote the scalar types of a tensor contraction to a common type.
"""
promote_contract(args...) = Base.promote_op(tensorop, args...)

"""
    promote_add(args...)

Promote the scalar types of a tensor addition to a common type.
"""
promote_add(args...) = Base.promote_op(+, args...)

"""
    tensoralloc_add(TC, A, pA, conjA, [istemp=Val(false), allocator])

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensoradd!(C, A, pA, conjA)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `allocator` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_add(
        TC, A, pA::Index2Tuple, conjA::Bool, istemp::Val = Val(false),
        allocator = DefaultAllocator()
    )
    ttype = tensoradd_type(TC, A, pA, conjA)
    structure = tensoradd_structure(A, pA, conjA)
    return tensoralloc(ttype, structure, istemp, allocator)
end

"""
    tensoralloc_contract(TC, A, pA, conjA, B, pB, conjB, pAB, [istemp=Val(false), allocator])

Allocate a tensor `C` of scalar type `TC` that would be the result of

    `tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)`

The `istemp` argument is used to indicate that a tensor wlil not be used after the `@tensor`
block, and thus will be followed by an explicit call to `tensorfree!`. The `allocator` can be
used to implement different allocation strategies.

See also [`tensoralloc`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc_contract(
        TC,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, istemp::Val = Val(false),
        allocator = DefaultAllocator()
    )
    ttype = tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)
    structure = tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)
    return tensoralloc(ttype, structure, istemp, allocator)
end

# ------------------------------------------------------------------------------------------
# AbstractArray implementation
# ------------------------------------------------------------------------------------------

tensorstructure(A::AbstractArray) = size(A)
tensorstructure(A::AbstractArray, iA::Int, conjA::Bool) = size(A, iA)

function tensoradd_type(TC, A::Array, pA::Index2Tuple, conjA::Bool)
    return Array{TC, sum(length.(pA))}
end
function tensoradd_type(TC, A::AbstractArray, pA::Index2Tuple, conjA::Bool)
    return Array{TC, sum(length.(pA))}
end
function tensoradd_type(TC, A::SubArray, pA::Index2Tuple, conjA::Bool)
    return tensoradd_type(TC, A.parent, pA, conjA)
end
function tensoradd_type(TC, A::Base.ReshapedArray, pA::Index2Tuple, conjA::Bool)
    return tensoradd_type(TC, A.parent, pA, conjA)
end
function tensoradd_type(TC, A::Base.PermutedDimsArray, pA::Index2Tuple, conjA::Bool)
    return tensoradd_type(TC, A.parent, pA, conjA)
end

function tensoradd_structure(A::AbstractArray, pA::Index2Tuple, conjA::Bool)
    return size.(Ref(A), linearize(pA))
end

function tensorcontract_type(
        TC,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple
    )
    T1 = tensoradd_type(TC, A, pAB, conjA)
    T2 = tensoradd_type(TC, B, pAB, conjB)
    if T1 == T2
        return T1
    else
        error("incompatible types for tensorcontract!: $T1 and $T2")
    end
end

function tensorcontract_structure(
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple
    )
    return let lA = length(pA[1])
        map(n -> n <= lA ? size(A, pA[1][n]) : size(B, pB[2][n - lA]), linearize(pAB))
    end
end

function tensoralloc(ttype, structure, ::Val = Val(false), allocator = DefaultAllocator())
    C = ttype(undef, structure)
    # fix an issue with undefined references for strided arrays
    if !isbitstype(scalartype(ttype))
        C = zerovector!!(C)
    end
    return C
end

tensorfree!(C, allocator = DefaultAllocator()) = nothing

# ------------------------------------------------------------------------------------------
# ManualAllocator implementation
# ------------------------------------------------------------------------------------------

function tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, ::ManualAllocator
    ) where {A <: AbstractArray, istemp}
    if istemp
        return malloc(eltype(A), structure...)
    else
        return tensoralloc(A, structure, Val(istemp))
    end
end

function tensorfree!(C::PtrArray, ::ManualAllocator)
    free(C)
    return nothing
end

# ------------------------------------------------------------------------------------------
# BufferAllocator implementation
# ------------------------------------------------------------------------------------------

# length in bytes
Base.length(buffer::BufferAllocator) = length(buffer.buffer)
Base.isempty(buffer::BufferAllocator) = iszero(buffer.offset)
Base.pointer(buffer::BufferAllocator, args...) = pointer(buffer.buffer, args...)

function Base.sizehint!(buffer::BufferAllocator, n::Integer; shrink::Bool = false)
    isempty(buffer) || error("Cannot resize a buffer that still contains elements")

    n = _buffersz(shrink ? n : max(n, length(buffer)))
    # not using resize! because might not be resizable
    (n == length(buffer)) || (buffer.buffer = similar(buffer.buffer, n))

    # reset max_offset if allowed to shrink
    shrink && (buffer.max_offset = 0)

    return buffer
end

# how many bytes should be reserved
allocation_size(::Type{T}, structure::Base.Dims) where {T} = prod(structure) * sizeof(T)

function tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, buffer::BufferAllocator
    ) where {A <: AbstractArray, istemp}
    if istemp
        T = eltype(A)
        offset = buffer.offset + allocation_size(T, structure)
        buffer.max_offset = max(buffer.max_offset, offset)

        # grow buffer if empty
        isempty(buffer) && sizehint!(buffer, buffer.max_offset)

        # Use pointer if there is enough space
        if offset < length(buffer)
            ptr = convert(Ptr{T}, pointer(buffer, buffer.offset))
            buffer.offset = offset
            return Base.unsafe_wrap(Array, ptr, structure)
        end
    end

    # Allocate default if not
    return A(undef, structure)
end

allocator_checkpoint!(buffer::BufferAllocator) = buffer.offset

function allocator_reset!(buffer::BufferAllocator, checkpoint)
    checkpoint ≤ buffer.offset ||
        throw(ArgumentError("Invalid checkpoint: `allocator_reset!` has to be called in reverse order on saved checkpoints"))
    buffer.offset = checkpoint
    return buffer
end
