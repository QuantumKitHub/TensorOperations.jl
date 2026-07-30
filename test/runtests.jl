using TensorOperations
using LinearAlgebra
using Test
using Random
Random.seed!(1234567)

using TensorOperations: IndexError
using TensorOperations: BaseCopy, BaseView, StridedNative, StridedBLAS
using TensorOperations: DefaultAllocator, ManualAllocator, BufferAllocator

precision(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
precision(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-8

# https://github.com/QuantumKitHub/TensorOperations.jl/issues/280: the generated code has to
# keep the user's `LineNumberNode`s -- Julia only emits a coverage counter for lines that a
# `LineNumberNode` points at -- and drop the ones synthesized by the parser, which would
# otherwise re-attribute the statements that follow them to a line in TensorOperations itself.
# Only `LineNumberNode`s in block-statement position matter for that: elsewhere (most notably
# the mandatory 2nd argument of a `:macrocall`) they are structurally required and left alone.
function statementlinenumbernodes(ex, acc = LineNumberNode[])
    if ex isa Expr
        if ex.head === :block
            for e in ex.args
                e isa LineNumberNode && push!(acc, e)
            end
        end
        foreach(e -> statementlinenumbernodes(e, acc), ex.args)
    end
    return acc
end

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
    @testset "allocator" verbose = true begin
        include("allocator.jl")
    end
    @testset "ad" verbose = false begin
        include("ad.jl")
    end
    @testset "mooncake" verbose = false begin
        include("mooncake.jl")
    end
    is_apple_ci = Sys.isapple() && get(ENV, "CI", "false") == "true"
    if !is_apple_ci
        @testset "enzyme" verbose = false begin
            include("enzyme.jl")
        end
    end
end

# note: cuTENSOR should not be loaded before this point
# as there is a test which requires it to be loaded after
@testset "cuTENSOR extension" verbose = true begin
    include("cutensor.jl")
end
@testset "GPUArrays" verbose = true begin
    include("gpu.jl")
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
