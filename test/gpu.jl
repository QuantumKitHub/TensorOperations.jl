using TensorOperations
using TensorOperations: StridedBLAS, StridedNative, linearize, numout
using Test
using Adapt
using TupleTools
using JLArrays
using VectorInterface
using CUDACore

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

backends = [StridedBLAS(), StridedNative()]

@testset "tensoradd! ($AT)" for AT in ATs
    sz = (3, 5, 4, 6)
    p = (3, 1, 4, 2)
    for backend in backends, T in (Float32, ComplexF32)
        A = randn(T, sz)
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

@testset "tensortrace! ($AT)" for AT in ATs
    sz = (2, 4, 3, 2)
    p = (2, 3)
    q = ((1,), (4,))

    for backend in backends, T in (Float32, ComplexF32)
        A = randn(T, sz)
        C = randn(T, TupleTools.getindices(sz, p))

        @test compare(AT, C, A) do c, a
            tensortrace!(c, a, (p, ()), q, false, One(), Zero(), backend)
        end

        α = rand(T)
        @test compare(AT, C, A) do c, a
            tensortrace!(c, a, (p, ()), q, false, α, Zero(), backend)
        end

        β = rand(T)
        @test compare(AT, C, A) do c, a
            tensortrace!(c, a, (p, ()), q, false, α, β, backend)
        end

        T <: Real || @test compare(AT, C, A) do c, a
            tensortrace!(c, a, (p, ()), q, true, α, β, backend)
        end
    end
end

@testset "tensorcontract! ($AT)" for AT in ATs
    sz = (2, 4, 3, 4, 2, 5)
    pA = ((4, 1), (2, 3))
    pB = ((3, 1), (2,))
    pAB = ((1, 2, 3), ())

    for backend in backends, T in (Float32, ComplexF32)
        A = randn(T, (2, 4, 3, 2))
        B = randn(T, (3, 3, 4))
        C = randn(T, (2, 2, 3))

        @test compare(AT, C, A, B) do c, a, b
            tensorcontract!(c, a, pA, false, b, pB, false, pAB, One(), Zero(), backend)
        end

        α = rand(T)
        @test compare(AT, C, A, B) do c, a, b
            tensorcontract!(c, a, pA, false, b, pB, false, pAB, α, Zero(), backend)
        end

        β = rand(T)
        @test compare(AT, C, A, B) do c, a, b
            tensorcontract!(c, a, pA, false, b, pB, false, pAB, α, β, backend)
        end

        if !(T <: Real)
            @test compare(AT, C, A, B) do c, a, b
                tensorcontract!(c, a, pA, true, b, pB, false, pAB, α, β, backend)
            end
            @test compare(AT, C, A, B) do c, a, b
                tensorcontract!(c, a, pA, false, b, pB, true, pAB, α, β, backend)
            end
            @test compare(AT, C, A, B) do c, a, b
                tensorcontract!(c, a, pA, true, b, pB, true, pAB, α, β, backend)
            end
        end
    end

end
