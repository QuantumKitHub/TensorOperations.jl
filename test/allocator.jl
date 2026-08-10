using TensorOperations
using TensorOperations: BufferAllocator, DefaultAllocator, ManualAllocator
using TensorOperations: JLBufferAllocator
using TensorOperations: tensoralloc, tensorfree!, tensoralloc_add, tensoralloc_contract
using TensorOperations: allocator_checkpoint!, allocator_reset!
using Test
using LinearAlgebra
using JLArrays

@testset "BufferAllocator" begin
    @testset "Constructor and basic properties" begin
        # Test default constructor
        buffer = BufferAllocator()
        @test buffer isa BufferAllocator
        @test length(buffer) == 0
        @test isempty(buffer)
        @test buffer.offset == 0
        @test buffer.max_offset == 0

        # Test constructor with sizehint
        buffer2 = BufferAllocator(; sizehint = 1024)
        @test length(buffer2) == 1024
        @test isempty(buffer2)
        @test buffer2.offset == 0
        @test buffer2.max_offset == 0

        # Test with explicit storage type
        buffer3 = BufferAllocator{Vector{UInt8}}(; sizehint = 512)
        @test buffer3 isa BufferAllocator{Vector{UInt8}}
        @test length(buffer3) >= 512

        # buffers smaller than a page are rounded up to a power of two (UInt8 elements)
        resize!(buffer, 3000)
        @test length(buffer) == 4096
        @test isempty(buffer)
        @test buffer.max_offset == 0

        # Cannot resize non-empty buffer
        buffer.offset = 100
        @test_throws ErrorException resize!(buffer, 4096)
        # Reset and try again
        empty!(buffer)
        resize!(buffer, 4096)
        @test length(buffer) == 4096

        # Test shrinking (only when allowed)
        sizehint!(buffer, 1024, shrink = true)
        @test buffer.max_offset == 1024

        # sizehint! does not shrink when shrink=false
        sizehint!(buffer, 512)
        @test buffer.max_offset == 1024
    end

    @testset "Buffer sizes" begin
        P = TensorOperations._pagesize()

        # below a page, sizes are rounded up to a power of two
        @test length(BufferAllocator(; sizehint = 0)) == 0
        @test length(BufferAllocator(; sizehint = 100)) == 128
        @test length(BufferAllocator(; sizehint = P)) == P

        # above a page, sizes are rounded up to a multiple of a page, and not all the way
        # up to the next power of two
        @test length(BufferAllocator(; sizehint = P + 1)) == 2P
        @test length(BufferAllocator(; sizehint = 10P + 1)) == 11P
        buffer = BufferAllocator(; sizehint = 4P + 123)
        @test length(buffer) == 5P
        resize!(buffer, 2^20 + 1)
        @test length(buffer) == cld(2^20 + 1, P) * P
        @test length(buffer) < 2^21

        # sizes are normalized to `Int`, whatever integer type they are computed from:
        # the storage constructors do not accept e.g. the `Int32` that `Clong` is on Windows
        @test TensorOperations._pagesize() isa Int
        @test TensorOperations._buffersz(Int32(100)) === 128
        @test TensorOperations._buffersz(UInt(10P + 1)) === 11P
        @test length(BufferAllocator(; sizehint = Int32(100))) == 128
        @test length(BufferAllocator(; sizehint = UInt(100))) == 128
    end

    @testset "buffer_arraytype" begin
        # a host buffer serves `Array`s at every rank, whatever its own storage is -- note that
        # `similar` of the default `Memory` storage stays a `Memory` at rank 1
        for buffer in (BufferAllocator(), BufferAllocator{Vector{UInt8}}(; sizehint = 1024))
            for A in (
                    Vector{Float64}, Matrix{ComplexF64}, Array{Float32, 3},
                    Vector{NTuple{4, Float64}},
                )
                @test TensorOperations.buffer_arraytype(A, buffer) === A
            end
            @static if isdefined(Core, :Memory)
                @test TensorOperations.buffer_arraytype(Memory{Float64}, buffer) === nothing
            end
        end
    end

    @testset "Checkpoint and reset" begin
        buffer = BufferAllocator(sizehint = 128)
        L = length(buffer)

        # Create checkpoint at beginning
        cp0 = allocator_checkpoint!(buffer)
        @test cp0 == 0

        # Allocate some tensors
        t1 = tensoralloc(Vector{UInt8}, 100, Val(true), buffer) # should fit
        @test t1 isa Vector{UInt8}
        @test size(t1) == (100,)
        cp1 = allocator_checkpoint!(buffer)
        @test cp1 > cp0
        # Verify pointer backing from buffer
        @test pointer(t1) == Ptr{UInt8}(pointer(buffer, cp0))
        @test buffer.offset == cp1
        @test buffer.max_offset == cp1

        # Allocate non-temporary tensor (should not use buffer)
        t3 = tensoralloc(Array{Float64, 2}, (10, 10), Val(false), buffer)
        @test t3 isa Array{Float64, 2}
        @test size(t3) == (10, 10)
        # offset should not change for non-temporary tensors
        @test buffer.offset == cp1 == buffer.max_offset

        t2 = tensoralloc(Array{Float32, 3}, (5, 5, 5), Val(true), buffer) # may not fit
        @test t2 isa Array{Float32, 3}
        @test size(t2) == (5, 5, 5)
        cp2 = allocator_checkpoint!(buffer)
        # buffer should have tracked required size, but offset only changes if it fit
        @test buffer.max_offset >= cp1

        # Reset to checkpoint 1
        allocator_reset!(buffer, cp1)
        @test buffer.offset == cp1
        @test length(buffer) == L # no auto resize on partial reset

        # reset to checkpoint in wrong order
        @test_throws ArgumentError allocator_reset!(buffer, cp1 + 10)

        # Reset to beginning, next allocation when empty auto-resizes
        allocator_reset!(buffer, cp0)
        @test isempty(buffer)

        # Trigger auto-resize on next temporary allocation
        tensoralloc(Array{UInt8, 2}, (L + 1, 1), Val(true), buffer)
        @test length(buffer) > L
    end

    @testset "ncon does not leak buffer space" begin
        buffer = BufferAllocator()
        A = randn(5, 5)
        B = randn(5, 5)
        C = randn(5, 5)
        # chain contraction A*B*C -> at least one intermediate tensor allocated as temp
        R = ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
        @test R ≈ A * B * C
        # offset must return to 0: intermediates were reclaimed via allocator_reset!
        @test buffer.offset == 0
        @test isempty(buffer)

        # repeated calls must not grow the high-water mark beyond a single call's needs
        max1 = buffer.max_offset
        for _ in 1:5
            ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
        end
        @test buffer.offset == 0
        @test buffer.max_offset == max1
    end
end

# `JLArrays` is the reference GPU array implementation, so a `JLArray`-backed buffer exercises
# the foreign-storage code paths of `BufferAllocator` -- the same ones that `CUDABufferAllocator`
# and `AMDBufferAllocator` rely on -- without requiring any GPU hardware.
@testset "JLArray-backed BufferAllocator" verbose = true begin
    # is the memory of `A` taken from `buffer`?
    function isbufferbacked(A, buffer)
        iszero(length(buffer)) && return false
        base = UInt(pointer(buffer))
        return base ≤ UInt(pointer(A)) < base + length(buffer)
    end

    @testset "Constructor and basic properties" begin
        buffer = JLBufferAllocator(; sizehint = 1024)
        @test buffer isa BufferAllocator{JLArray{UInt8, 1}}
        @test buffer isa typeof(BufferAllocator{JLArray{UInt8, 1}}(; sizehint = 1024))
        @test length(buffer) == 1024
        @test isempty(buffer)
        @test buffer.offset == 0

        # resizing frees the old buffer and allocates a new one
        resize!(buffer, 3000)
        @test length(buffer) == 4096
        @test isempty(buffer)
        buffer.offset = 100
        @test_throws ErrorException resize!(buffer, 8192)
        empty!(buffer)
        @test length(resize!(buffer, 8192)) == 8192
    end

    @testset "tensoralloc" begin
        buffer = JLBufferAllocator(; sizehint = 4096)

        # temporaries are taken from the buffer
        C1 = tensoralloc(JLArray{Float32, 2}, (8, 8), Val(true), buffer)
        @test C1 isa JLArray{Float32, 2}
        @test size(C1) == (8, 8)
        @test isbufferbacked(C1, buffer)
        @test buffer.offset == 8 * 8 * sizeof(Float32)

        # non-temporaries are not
        offset = buffer.offset
        C2 = tensoralloc(JLArray{Float32, 2}, (8, 8), Val(false), buffer)
        @test C2 isa JLArray{Float32, 2}
        @test !isbufferbacked(C2, buffer)
        @test buffer.offset == offset

        # freeing a buffer-backed tensor does not invalidate the buffer: `unsafe_buffer_wrap`
        # only retains a reference to it, so freeing merely releases that reference again
        ptr1 = pointer(C1)
        tensorfree!(C1, buffer)
        allocator_reset!(buffer, 0)
        C3 = tensoralloc(JLArray{Float32, 2}, (8, 8), Val(true), buffer)
        @test isbufferbacked(C3, buffer)
        @test pointer(C3) == ptr1
        fill!(C3, 1.0f0)
        @test all(isone, collect(C3))

        # a bare length is accepted for vectors
        allocator_reset!(buffer, 0)
        C4 = tensoralloc(JLArray{Float64, 1}, 16, Val(true), buffer)
        @test C4 isa JLArray{Float64, 1}
        @test size(C4) == (16,)
        @test isbufferbacked(C4, buffer)
    end

    @testset "storage mismatch falls back" begin
        # a host buffer cannot back JLArrays
        hostbuffer = BufferAllocator(; sizehint = 4096)
        C1 = tensoralloc(JLArray{Float32, 2}, (8, 8), Val(true), hostbuffer)
        @test C1 isa JLArray{Float32, 2}
        @test hostbuffer.offset == 0

        # a JLArray buffer cannot back Arrays
        jlbuffer = JLBufferAllocator(; sizehint = 4096)
        C2 = tensoralloc(Array{Float64, 2}, (8, 8), Val(true), jlbuffer)
        @test C2 isa Array{Float64, 2}
        @test jlbuffer.offset == 0
    end

    @testset "alignment" begin
        buffer = JLBufferAllocator(; sizehint = 8192)
        @test TensorOperations.buffer_alignment(buffer) == 16
        @test iszero(UInt(pointer(buffer)) % 16)

        # a deliberately misaligning allocation of 3 bytes
        C1 = tensoralloc(JLArray{UInt8, 1}, (3,), Val(true), buffer)
        @test isbufferbacked(C1, buffer)
        @test buffer.offset == 3
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            C2 = tensoralloc(JLArray{T, 1}, (4,), Val(true), buffer)
            @test isbufferbacked(C2, buffer)
            @test iszero(UInt(pointer(C2)) % 16)
        end

        # `GPUArrays.derive` takes an element offset, so the padding of an element type whose
        # size does not divide the alignment is a multiple of that size instead: such types are
        # still served from the buffer, at an offset that survives the conversion to elements
        # rather than silently truncating onto the previous temporary
        @test TensorOperations.buffer_arraytype(JLArray{ComplexF64, 1}, buffer) ===
            JLArray{ComplexF64, 1}
        @test TensorOperations.buffer_arraytype(JLArray{NTuple{4, Float64}, 1}, buffer) ===
            JLArray{NTuple{4, Float64}, 1}
        for T in (NTuple{4, Float64}, NTuple{3, Float32})
            empty!(buffer)
            C3 = tensoralloc(JLArray{UInt8, 1}, (3,), Val(true), buffer)
            C4 = tensoralloc(JLArray{T, 1}, (4,), Val(true), buffer)
            @test isbufferbacked(C4, buffer)
            start = UInt(pointer(C4)) - UInt(pointer(buffer))
            @test iszero(start % sizeof(T))
            @test iszero(start % TensorOperations.buffer_alignment(buffer))
            # no overlap with the 3 bytes that `C3` occupies
            @test start ≥ 3
            @test buffer.offset == start + 4 * sizeof(T)
        end
    end

    @testset "checkpoint and reset" begin
        buffer = JLBufferAllocator(; sizehint = 4096)
        cp0 = allocator_checkpoint!(buffer)
        @test cp0 == 0

        C1 = tensoralloc(JLArray{Float32, 2}, (8, 8), Val(true), buffer)
        cp1 = allocator_checkpoint!(buffer)
        @test cp1 > cp0
        C2 = tensoralloc(JLArray{Float32, 2}, (8, 8), Val(true), buffer)
        @test pointer(C2) != pointer(C1)

        allocator_reset!(buffer, cp1)
        @test buffer.offset == cp1
        @test_throws ArgumentError allocator_reset!(buffer, cp1 + 10)

        allocator_reset!(buffer, cp0)
        @test isempty(buffer)
    end

    @testset "tensor network ($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
        D1, D2, D3 = 30, 40, 20
        d1, d2 = 2, 3

        A1 = JLArray(randn(T, D1, d1, D2))
        A2 = JLArray(randn(T, D2, d2, D3))
        ρₗ = JLArray(randn(T, D1, D1))
        ρᵣ = JLArray(randn(T, D3, D3))
        H = JLArray(randn(T, d1, d2, d1, d2))

        @tensor begin
            HRAA1[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                ρᵣ[c', c] * H[s1, s2, t1, t2]
        end

        buffer = JLBufferAllocator()
        @tensor allocator = buffer begin
            HRAA2[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                ρᵣ[c', c] * H[s1, s2, t1, t2]
        end
        @test HRAA2 isa JLArray{T, 4}
        @test collect(HRAA2) ≈ collect(HRAA1)

        # all temporaries were reclaimed, and the buffer was actually used
        @test buffer.offset == 0
        @test buffer.max_offset > 0

        # The high-water mark only counts the temporaries that actually fit in the buffer, so it
        # may still grow while the buffer is warming up, but it has to converge to a fixed size
        # after a couple of contractions.
        max0 = buffer.max_offset
        for _ in 1:5
            @tensor allocator = buffer begin
                HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @test collect(HRAA3) ≈ collect(HRAA1)
        end
        max1 = buffer.max_offset
        @test max1 ≥ max0
        @test length(buffer) ≥ max1

        for _ in 1:5
            @tensor allocator = buffer begin
                HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @test collect(HRAA3) ≈ collect(HRAA1)
        end
        @test buffer.offset == 0
        @test buffer.max_offset == max1

        # scalar output
        @tensor begin
            E1 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] *
                H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c'])
        end
        @tensor allocator = buffer begin
            E2 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] *
                H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c'])
        end
        @test E1 ≈ E2
        @test buffer.offset == 0
    end

    @testset "ncon" begin
        A = JLArray(randn(Float32, 5, 5))
        B = JLArray(randn(Float32, 5, 5))
        C = JLArray(randn(Float32, 5, 5))
        buffer = JLBufferAllocator()

        R = ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
        @test R isa JLArray{Float32, 2}
        @test collect(R) ≈ collect(A) * collect(B) * collect(C)
        @test buffer.offset == 0

        max1 = buffer.max_offset
        for _ in 1:5
            ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
        end
        @test buffer.offset == 0
        @test buffer.max_offset == max1
    end
end
