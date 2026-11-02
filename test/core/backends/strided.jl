using TensorOperations
using TensorOperations: StridedBLAS, StridedNative, ManualAllocator, BufferAllocator
using Test

@isdefined(TestSuite) || include("../../testsuite/TestSuite.jl")

@testset "StridedNative" begin
    TestSuite.test_backend(StridedNative(); diagonal = true, aliasing = true)
end
@testset "StridedBLAS" begin
    TestSuite.test_backend(StridedBLAS(); aliasing = true)
end
@testset "$backend with generic element types" for backend in (StridedNative(), StridedBLAS())
    TestSuite.test_backend(backend; eltypes = (BigFloat, Complex{BigFloat}))
end
@testset "$backend with allocator $allocator" for backend in (StridedNative(), StridedBLAS()),
        allocator in (ManualAllocator(), BufferAllocator())

    TestSuite.test_tensorcontract(backend; allocator, eltypes = (Float64, ComplexF64))
end
