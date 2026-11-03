using TensorOperations
using TensorOperations: BaseView
using Test

@isdefined(TestSuite) || include("../../testsuite/TestSuite.jl")

@testset "BaseView" begin
    TestSuite.test_backend(BaseView(); diagonal = true)
end
