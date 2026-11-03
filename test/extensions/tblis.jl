using TensorOperations
using TensorOperations: TBLISBackend, StridedNative
using TBLIS
using LinearAlgebra
using Random
using Test

Random.seed!(1234567)

const eltypes = (Float32, Float64, ComplexF32, ComplexF64)
const tblis = TBLISBackend()
const reference = StridedNative()

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")

# TBLIS requires all arguments to have the same element type
@testset "test suite" begin
    TestSuite.test_backend(tblis; eltypes, mixed_eltypes = (), aliasing = true)
end

@testset "rejection of unsupported arguments" begin
    # element types TBLIS does not know about
    for A in (
            randn(Float16, (3, 4)), rand(1:4, (3, 4)),
            Rational{Int}.(rand(1:4, (3, 4)), 3),
        )
        T = eltype(A)
        B = T <: AbstractFloat ? randn(T, (4, 5)) : T.(rand(1:4, (4, 5)))
        C = zeros(T, (3, 5))
        @test_throws ArgumentError tensorcontract!(
            C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1, 0, tblis
        )
        @test_throws ArgumentError tensoradd!(
            zeros(T, (4, 3)), A, ((2, 1), ()), false, 1, 0, tblis
        )
    end

    # mixed element types: TBLIS requires a single type for all tensors
    A = randn(Float64, (3, 4))
    B = randn(ComplexF64, (4, 5))
    C = zeros(ComplexF64, (3, 5))
    @test_throws ArgumentError tensorcontract!(
        C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1, 0, tblis
    )

    # non-strided arrays
    D = Diagonal(randn(Float64, 4))
    A = randn(Float64, (3, 4))
    C = zeros(Float64, (3, 4))
    @test_throws ArgumentError tensorcontract!(
        C, A, ((1,), (2,)), false, D, ((1,), (2,)), false, ((1, 2), ()), 1, 0, tblis
    )

    # writing into a conjugated view would silently produce the conjugate of the result
    A = randn(ComplexF64, (3, 4))
    Cadj = adjoint(zeros(ComplexF64, (4, 3)))
    @test_throws ArgumentError tensoradd!(
        Cadj, A, ((1, 2), ()), false, 1, 0, tblis
    )
end

@testset "@tensor and ncon integration (eltype = $T)" for T in eltypes
    A = randn(T, (5, 5, 5, 5))
    B = randn(T, (5, 5, 5))
    C = randn(T, (5, 5, 5))

    @tensor backend = tblis D[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * conj(C[g, f, b])
    @tensor backend = reference Dref[a, b, c, d] := A[a, e, c, f] * B[g, d, e] *
        conj(C[g, f, b])
    @test D ≈ Dref

    network = [[-1, 1, -3, 2], [3, -4, 1], [3, 2, -2]]
    conjlist = [false, false, true]
    @test ncon([A, B, C], network, conjlist; backend = tblis) ≈ Dref
    @test ncon([A, B, C], network; backend = tblis) ≈
        ncon([A, B, C], network; backend = reference)

    # traces and scalar results
    @tensor backend = tblis s = A[a, b, a, b]
    @tensor backend = reference sref = A[a, b, a, b]
    @test s ≈ sref
end

@testset "garbage collection safety" begin
    # A `tblis_tensor` only stores raw pointers into the array and into the buffers holding
    # its lengths and strides; make sure nothing goes missing under GC pressure.
    T = ComplexF64
    A = randn(T, (12, 8, 6))
    B = randn(T, (8, 6, 10))
    Cref = similar(A, (12, 10))
    tensorcontract!(
        Cref, A, ((1,), (2, 3)), true, B, ((1, 2), (3,)), false, ((1, 2), ()), 1, 0,
        reference
    )
    for i in 1:100
        C = similar(A, (12, 10))
        tensorcontract!(
            C, A, ((1,), (2, 3)), true, B, ((1, 2), (3,)), false, ((1, 2), ()), 1, 0, tblis
        )
        @test C ≈ Cref
        iszero(i % 10) && GC.gc(true)
        # keep the allocator busy so that freed buffers get reused promptly
        junk = [randn(T, 1024) for _ in 1:8]
        @test length(junk) == 8
    end
end

@testset "threading" begin
    nthreads = TBLIS.get_num_threads()
    counts = ntuple(identity, min(2, nthreads))
    try
        A = randn(Float64, (40, 40, 20))
        B = randn(Float64, (20, 40, 40))
        Cref = similar(A, (40, 40, 40, 40))
        tensorcontract!(
            Cref, A, ((1, 2), (3,)), false, B, ((1,), (2, 3)), false,
            ((1, 2, 3, 4), ()), 1, 0, reference
        )
        for n in counts
            TBLIS.set_num_threads(n)
            @test TBLIS.get_num_threads() == n
            C = similar(A, (40, 40, 40, 40))
            tensorcontract!(
                C, A, ((1, 2), (3,)), false, B, ((1,), (2, 3)), false,
                ((1, 2, 3, 4), ()), 1, 0, tblis
            )
            @test C ≈ Cref
        end
    finally
        TBLIS.set_num_threads(nthreads)
    end
end
