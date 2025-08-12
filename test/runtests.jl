using TensorOperations
using LinearAlgebra
using Test
using Random
Random.seed!(1234567)

using TensorOperations: IndexError
using TensorOperations: BaseCopy, BaseView, StridedNative, StridedBLAS
using TensorOperations: DefaultAllocator, ManualAllocator

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
