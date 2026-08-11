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
    AMDAllocator()

Allocator that uses the AMD memory manager and will thus allocate `ROCArray` instances.
"""
struct AMDAllocator end

"""
    CUDABufferAllocator(; sizehint = 0, memory = CUDA.default_memory)

Convenience constructor for a [`BufferAllocator`](@ref) that is backed by CUDA memory, and
which will thus hand out `CuArray` instances that are carved out of a single pre-allocated
buffer. The `memory` keyword can be any of the CUDA.jl memory types, i.e.
`CUDA.DeviceMemory`, `CUDA.UnifiedMemory` or `CUDA.HostMemory`, and determines both where the
buffer itself lives and in which memory space the temporary tensors will be located.

This requires `CUDACore` to be loaded, and is equivalent to spelling out the storage type as
`BufferAllocator{CuArray{UInt8, 1, memory}}(; sizehint)`.

See also [`TensorOperations.BufferAllocator`](@ref) and [`TensorOperations.CUDAAllocator`](@ref).
"""
function CUDABufferAllocator end

"""
    AMDBufferAllocator(; sizehint = 0, buftype = AMDGPU.Mem.HIPBuffer)

Convenience constructor for a [`BufferAllocator`](@ref) that is backed by AMD memory, and which
will thus hand out `ROCArray` instances that are carved out of a single pre-allocated buffer.
The `buftype` keyword can be any of the AMDGPU.jl buffer types, i.e.
`AMDGPU.Mem.HIPBuffer` or `AMDGPU.Mem.HostBuffer`, and determines both where the buffer itself
lives and in which memory space the temporary tensors will be located.

This requires `AMDGPU` to be loaded, and is equivalent to spelling out the storage type as
`BufferAllocator{ROCArray{UInt8, 1, buftype}}(; sizehint)`.

See also [`TensorOperations.BufferAllocator`](@ref) and [`TensorOperations.AMDAllocator`](@ref).
"""
function AMDBufferAllocator end

"""
    JLBufferAllocator(; sizehint = 0)

Convenience constructor for a [`BufferAllocator`](@ref) that is backed by a `JLArray`, and which
will thus hand out `JLArray` instances that are carved out of a single pre-allocated buffer.
As `JLArrays` is the reference GPU array implementation, this is mostly useful for testing the
foreign-storage code paths of `BufferAllocator` without requiring actual GPU hardware.

This requires `JLArrays` to be loaded, and is equivalent to spelling out the storage type as
`BufferAllocator{JLArray{UInt8, 1}}(; sizehint)`.

See also [`TensorOperations.BufferAllocator`](@ref).
"""
function JLBufferAllocator end

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
    BufferAllocator{Storage}(; sizehint = 0)

Allocator that uses a pre-allocated buffer for storing temporary tensors.
When the buffer is full, the allocator falls back on Julia's default allocation mechanism
to create temporary tensors, but keeps track of how much additional memory is required.
When the buffer is fully reset, the buffer is automatically resized to ensure subsequent
contractions will now fit in the buffer.

The optional type parameter `Storage` determines the container that backs the buffer, and
must have single-byte elements. It defaults to `Memory{UInt8}` (or `Vector{UInt8}` on Julia
versions without `Memory`), which hands out regular `Array` temporaries. Other storage types
can be supported by implementing [`TensorOperations.unsafe_buffer_wrap`](@ref);
in particular, `CuArray`-, `ROCArray`- and
`JLArray`-backed buffers are supported through [`TensorOperations.CUDABufferAllocator`](@ref),
[`TensorOperations.AMDBufferAllocator`](@ref) and [`TensorOperations.JLBufferAllocator`](@ref).

!!! warning
    This allocator is **not** thread-safe, and it is the user's responsibility to avoid running
    the same allocator on concurrent jobs. For concurrent usage, it is recommended to either
    manually use a separate buffer per task, or leverage Bumper.jl through [`@butensor`](@ref)
    instead.
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

# storages whose `pointer` is a host pointer, so that an `Array` can be wrapped around it
const HostStorageType = @static isdefined(Core, :Memory) ? Union{Memory, Array} : Array

# `Sys.PAGESIZE` only exists on sufficiently recent Julia versions; fall back on the standard
# page size otherwise, as this only serves as a granularity for rounding buffer sizes.
# Note the conversion to `Int`: the underlying `Clong` is 32 bits wide on Windows.
@static if isdefined(Sys, :PAGESIZE)
    _pagesize() = Int(Sys.PAGESIZE)
else
    _pagesize() = 4096
end

# Allocate buffers in sizes that are a multiple of the page size.
# Below a single page, powers of two are used instead, as rounding every small buffer up to a full page would be wasteful.
# The result is always an `Int`, as that is what the storage constructors expect.
function _buffersz(x::Integer)
    iszero(x) && return 0
    pagesize = _pagesize()
    x ≤ pagesize && return Int(Base.nextpow(2, x))
    return Int(cld(x, pagesize) * pagesize)
end

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
function tensoradd_type(TC, A::StridedView, pA::Index2Tuple, conjA::Bool)
    return tensoradd_type(TC, parent(A), pA, conjA)
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
Base.pointer(buffer::BufferAllocator) = pointer(buffer.buffer)
Base.pointer(buffer::BufferAllocator, offset) = pointer(buffer) + offset

Base.empty!(buffer::BufferAllocator) = (buffer.offset = 0; buffer)

function Base.resize!(buffer::BufferAllocator, n::Integer)
    isempty(buffer) || error("Cannot resize a buffer that still contains elements")
    n = _buffersz(n)
    n == length(buffer) || (buffer.buffer = similar(buffer.buffer, n))
    return buffer
end
function Base.resize!(buffer::BufferAllocator{<:Vector}, n::Integer)
    isempty(buffer) || error("Cannot resize a buffer that still contains elements")
    n = _buffersz(n)
    n == length(buffer) || resize!(buffer.buffer, n)
    return buffer
end

function Base.sizehint!(buffer::BufferAllocator, n::Integer; shrink::Bool = false)
    buffer.max_offset = shrink ? n : max(buffer.max_offset, n)
    return buffer
end

# how many bytes should be reserved
allocation_size(::Type{T}, structure::Base.Dims) where {T} = prod(structure) * sizeof(T)
allocation_size(::Type{T}, structure::Int) where {T} = structure * sizeof(T)

"""
    buffer_alignment(buffer::BufferAllocator)

The minimum alignment, in bytes, to which the temporaries handed out by `buffer` are padded.
This has to be a power of two, and currently there is no point in making it larger
than the alignment of the buffer's own base pointer, as the padding would then not
actually buy any alignment. Element types whose size is not a divisor of it are padded
further, to a multiple of that size as well.

Defaults to `16`, which is the alignment that Julia guarantees for its allocations,
and which covers the natural alignment of all standard element types.

See also [`TensorOperations.buffer_arraytype`](@ref).
"""
buffer_alignment(::BufferAllocator) = 16

# round `offset` up to the next multiple of `alignment`
function _alignup(offset::Integer, alignment::Integer)
    a = oftype(offset, alignment)
    # the bit trick only holds for powers of two, which `_buffer_alignment` is free not to be
    ispow2(a) && return (offset + a - one(a)) & ~(a - one(a))
    return cld(offset, a) * a
end

# The alignment `tensoralloc` pads a temporary of element type `T` to. The multiple of
# `sizeof(T)` keeps the offset expressible in elements, which is how some backends carry it,
# and costs no additional padding for element types whose size divides the alignment.
function _buffer_alignment(::Type{T}, buffer::BufferAllocator) where {T}
    alignment = max(Base.datatype_alignment(T), buffer_alignment(buffer))
    return iszero(sizeof(T)) ? alignment : lcm(sizeof(T), alignment)
end

# `structure` is a shape for arrays, but a bare length is accepted for vectors
_asdims(structure::Base.Dims) = structure
_asdims(n::Integer) = (Int(n),)

"""
    buffer_arraytype(::Type{A}, buffer::BufferAllocator)

Return the concrete array type that is used to serve a temporary allocation of type `A` from
`buffer`, or `nothing` if `buffer` cannot back arrays of type `A`, in which case the regular
allocation path is used instead. This only depends on the types involved, such that the
choice is resolved at compile time.

The default answer is inferred from [`TensorOperations.unsafe_buffer_wrap`](@ref): the type
that wrapping `buffer`'s memory actually produces, or `nothing` if that is not a concrete
subtype of `A`, which includes the case where no method applies. A storage therefore only has
to implement `unsafe_buffer_wrap` for its arrays to be served from the buffer, and one whose
wrap is not inferrable loses buffer backing rather than becoming incorrect.
"""
function buffer_arraytype(::Type{A}, buffer::BufferAllocator) where {A <: AbstractArray}
    S = Base.promote_op(
        unsafe_buffer_wrap, Type{A}, typeof(buffer), Int, Base.Dims{ndims(A)}
    )
    return (isconcretetype(S) && S <: A) ? S : nothing
end

"""
    unsafe_buffer_wrap(::Type{A}, buffer::BufferAllocator, start, structure) -> A

Wrap the memory of `buffer`, starting at byte offset `start`, into an array of type `A` with
shape `structure`. Here, `A` is the type returned by
[`TensorOperations.buffer_arraytype`](@ref), and it is the caller's responsibility to ensure
that the requested range actually fits within the buffer.

`start` is guaranteed to be a multiple of `sizeof(eltype(A))`, so arrays that carry an element
offset rather than a pointer can use `start ÷ sizeof(eltype(A))` without losing bytes.
"""
function unsafe_buffer_wrap(
        ::Type{A}, buffer::BufferAllocator{<:HostStorageType}, start, structure
    ) where {A <: Array}
    ptr = convert(Ptr{eltype(A)}, pointer(buffer, start))
    return Base.unsafe_wrap(Array, ptr, structure)
end

function tensoralloc(
        ::Type{A}, structure, ::Val{istemp}, buffer::BufferAllocator
    ) where {A <: AbstractArray, istemp}
    AA = buffer_arraytype(A, buffer)
    if istemp && AA !== nothing
        T = eltype(AA)
        nbytes = allocation_size(T, structure)
        if !iszero(nbytes) # empty temporaries have no meaningful pointer
            start = _alignup(buffer.offset, _buffer_alignment(T, buffer))
            offset = start + nbytes
            sizehint!(buffer, offset)

            # grow buffer if empty: this should never shrink the buffer, as that would
            # discard the size that was requested through `sizehint` or `resize!`
            if isempty(buffer) && buffer.max_offset > length(buffer)
                resize!(buffer, buffer.max_offset)
            end

            # Use pointer if there is enough space
            if offset <= length(buffer)
                buffer.offset = offset
                return unsafe_buffer_wrap(AA, buffer, start, structure)
            end
        end

        # Allocate in the same memory space as the buffer if it does not fit
        return AA(undef, structure)
    end

    # Allocate default if the buffer cannot back this type of array
    return A(undef, structure)
end

allocator_checkpoint!(buffer::BufferAllocator) = buffer.offset

function allocator_reset!(buffer::BufferAllocator, checkpoint)
    checkpoint ≤ buffer.offset ||
        throw(ArgumentError("Invalid checkpoint: `allocator_reset!` has to be called in reverse order on saved checkpoints"))
    buffer.offset = checkpoint
    return buffer
end
