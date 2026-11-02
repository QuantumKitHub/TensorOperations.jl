using TensorOperations
using TensorOperations: StridedBLAS, StridedNative, DefaultBackend, cuTENSORBackend, linearize, numout
using Test
using Adapt
using TupleTools
using JLArrays
using VectorInterface
using CUDACore, AMDGPU
using cuTENSOR # contractions of `CuArray`s require the cuTENSOR extension
using cuBLAS # matrix multiplications of `CuArray`s with `StridedBLAS` require the cuBLAS extension of Strided

test_result(a::AbstractArray, b::AbstractArray; kwargs...) =
    isapprox(collect(a), collect(b); kwargs...)

function compare(f, AT::Type, xs...; kwargs...)
    cpu_in = map(deepcopy, xs) # copy on CPU
    gpu_in = map(adapt(AT), xs) # adapt on GPU

    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)

    return test_result(cpu_out, gpu_out; kwargs...)
end

# types to test for
ATs = []
!is_buildkite && push!(ATs, JLArray)
CUDACore.functional() && push!(ATs, CuArray)
AMDGPU.functional() && push!(ATs, ROCArray)

backends = [StridedBLAS(), StridedNative()]

# storage-specific `BufferAllocator` constructor for each of the array types above
bufferallocator(::Type{JLArray}; kwargs...) = TensorOperations.JLBufferAllocator(; kwargs...)
bufferallocator(::Type{CuArray}; kwargs...) = TensorOperations.CUDABufferAllocator(; kwargs...)
bufferallocator(::Type{ROCArray}; kwargs...) = TensorOperations.AMDBufferAllocator(; kwargs...)

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")

@testset "test suite: $backend ($AT)" for AT in ATs, backend in backends
    TestSuite.test_backend(backend; arraytype = AT, eltypes = (Float32, ComplexF32))
end

# cuTENSOR requires all arguments to have the same element type, and positive strides; the
# default backend selects cuTENSOR only for equal element types, and Strided otherwise
if CUDACore.functional()
    @testset "test suite: cuTENSORBackend() (CuArray)" begin
        TestSuite.test_backend(
            cuTENSORBackend(); arraytype = CuArray, eltypes = (Float32, ComplexF32),
            mixed_eltypes = (), negative_strides = false
        )
    end
    @testset "test suite: DefaultBackend() (CuArray)" begin
        TestSuite.test_backend(
            DefaultBackend(); arraytype = CuArray, eltypes = (Float32, ComplexF32),
            negative_strides = false
        )
    end
end

@testset "tensoradd! with Diagonal ($AT)" for AT in ATs
    sz = (8, 8)
    p = (2, 1)
    diag_backends = [BaseCopy(), BaseView()]
    for backend in diag_backends, T in (Float32, ComplexF32)
        A = Diagonal(randn(T, sz[1]))
        C = randn(T, TupleTools.getindices(sz, p))

        @test compare(AT, C, A) do c, a
            tensoradd!(c, a, (p, ()), false, One(), Zero(), backend)
        end

        α = rand(T)
        @test compare(AT, C, A) do c, a
            tensoradd!(c, a, (p, ()), false, α, Zero(), backend)
        end

        β = rand(T)
        @test compare(AT, C, A) do c, a
            tensoradd!(c, a, (p, ()), false, α, β, backend)
        end

        T <: Real || @test compare(AT, C, A) do c, a
            tensoradd!(c, a, (p, ()), true, α, β, backend)
        end
    end
end

@testset "BufferAllocator ($AT)" for AT in ATs
    @testset "tensor network ($T)" for T in (Float32, ComplexF32)
        D1, D2, D3 = 30, 40, 20
        d1, d2 = 2, 3

        A1 = adapt(AT, randn(T, D1, d1, D2))
        A2 = adapt(AT, randn(T, D2, d2, D3))
        ρₗ = adapt(AT, randn(T, D1, D1))
        ρᵣ = adapt(AT, randn(T, D3, D3))
        H = adapt(AT, randn(T, d1, d2, d1, d2))

        @tensor begin
            HRAA1[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                ρᵣ[c', c] * H[s1, s2, t1, t2]
        end

        buffer = bufferallocator(AT)
        @tensor allocator = buffer begin
            HRAA2[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                ρᵣ[c', c] * H[s1, s2, t1, t2]
        end
        @test HRAA2 isa AT{T, 4}
        @test test_result(HRAA2, HRAA1)

        # all temporaries are reclaimed, and the buffer is actually used
        @test buffer.offset == 0
        @test buffer.max_offset > 0

        if AT === JLArray
            # Disjoint JLArray slices are reported as aliases once the buffer is warm.
            # https://github.com/JuliaGPU/GPUArrays.jl/pull/803
            # https://github.com/QuantumKitHub/StridedViews.jl/pull/57
            @test_broken begin
                @tensor allocator = buffer begin
                    HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                        ρᵣ[c', c] * H[s1, s2, t1, t2]
                end
                test_result(HRAA3, HRAA1)
            end
            continue # Later checks require the warm-buffer contraction to succeed.
        end

        # the high-water mark only counts temporaries that actually fit, so it may still grow
        # while the buffer is warming up, but has to converge to a fixed size afterwards
        for _ in 1:3
            @tensor allocator = buffer begin
                HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @test test_result(HRAA3, HRAA1)
        end
        max1 = buffer.max_offset
        for _ in 1:3
            @tensor allocator = buffer begin
                HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
        end
        @test buffer.offset == 0
        @test buffer.max_offset == max1
        @test length(buffer) ≥ max1

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

    # chain contraction through `ncon`, which reclaims its temporaries via checkpoints
    @testset "ncon" begin
        A = adapt(AT, randn(Float32, 5, 5))
        B = adapt(AT, randn(Float32, 5, 5))
        C = adapt(AT, randn(Float32, 5, 5))
        buffer = bufferallocator(AT)

        R = ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
        @test R isa AT{Float32, 2}
        @test test_result(R, collect(A) * collect(B) * collect(C))
        @test buffer.offset == 0

        max1 = buffer.max_offset
        for _ in 1:3
            ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
        end
        @test buffer.offset == 0
        @test buffer.max_offset == max1
    end
end
