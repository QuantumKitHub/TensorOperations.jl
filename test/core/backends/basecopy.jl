using TensorOperations
using TensorOperations: BaseCopy
using Test

@isdefined(TestSuite) || include("../../testsuite/TestSuite.jl")

# `BaseCopy` is the reference for all other backends, and is itself checked against loops
@testset "BaseCopy" begin
    TestSuite.test_backend(BaseCopy(); reference = TestSuite.LoopReference(), diagonal = true)
end
@testset "BaseCopy with generic element types" begin
    TestSuite.test_backend(
        BaseCopy(); reference = TestSuite.LoopReference(), eltypes = (BigFloat, Complex{BigFloat})
    )
end
