using TensorOperations
using LinearAlgebra
using Test
using Random
Random.seed!(1234567)

using TensorOperations: IndexError
using TensorOperations: BaseCopy, BaseView, StridedNative, StridedBLAS
using TensorOperations: DefaultAllocator, ManualAllocator

precision(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
precision(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-8

# don't run all tests on GPU, only the GPU
# specific ones
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
if !is_buildkite
    @testset "tensoropt" verbose = true begin
        include("tensoropt.jl")
    end
    @testset "auxiliary" verbose = true begin
        include("auxiliary.jl")
    end
    @testset "macro keywords" verbose = true begin
        include("macro_kwargs.jl")
    end
    @testset "method syntax" verbose = true begin
        include("methods.jl")
    end
    @testset "macro with index notation" verbose = true begin
        include("tensor.jl")
    end
    @testset "ad" verbose = false begin
        include("ad.jl")
    end
    @testset "mooncake" verbose = false begin
        include("mooncake.jl")
    end
    # mystery segfault on 1.10 for now
    @static if VERSION >= v"1.11.0"
        @testset "enzyme" verbose = false begin
            include("enzyme.jl")
        end
    end
end

if is_buildkite
    # note: cuTENSOR should not be loaded before this point
    # as there is a test which requires it to be loaded after
    @testset "cuTENSOR extension" verbose = true begin
        include("cutensor.jl")
    end
end

if !is_buildkite
    # note: Bumper should not be loaded before this point
    # as there is a test which requires it to be loaded after
    @testset "Bumper extension" verbose = true begin
        include("butensor.jl")
    end

    @testset "Polynomials" begin
        include("polynomials.jl")
    end

    @testset "Aqua" verbose = true begin
        using Aqua
        Aqua.test_all(TensorOperations)
    end
end
