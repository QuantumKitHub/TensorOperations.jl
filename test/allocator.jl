using TensorOperations
using TensorOperations: BufferAllocator, DefaultAllocator, ManualAllocator
using TensorOperations: tensoralloc, tensorfree!, tensoralloc_add, tensoralloc_contract
using TensorOperations: allocator_checkpoint!, allocator_reset!
using Test
using LinearAlgebra

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

        # sizehint! grows to next power-of-two elements (UInt8)
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

    @testset "Checkpoint and reset" begin
        buffer = BufferAllocator(sizehint = 128)
        L = length(buffer)

        # Create checkpoint at beginning
        cp0 = allocator_checkpoint!(buffer)
        @test cp0 == 0

        # Allocate some tensors
        t1 = tensoralloc(Array{UInt8, 2}, (10, 10), Val(true), buffer) # should fit
        @test t1 isa Array{UInt8, 2}
        @test size(t1) == (10, 10)
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
end
