using TensorOperations
using TensorOperations: DefaultBackend
using Test

@isdefined(TestSuite) || include("../../testsuite/TestSuite.jl")

@testset "DefaultBackend" begin
    TestSuite.test_backend(DefaultBackend(); diagonal = true)
end
