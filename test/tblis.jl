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

# `Zero()`/`One()` and plain numbers should all work; poison the output with `NaN` whenever
# `β == 0` so that a kernel that computes `0 * C` instead of ignoring `C` is caught.
poison!(C) = fill!(C, convert(eltype(C), NaN))

@testset "tensoradd! (eltype = $T)" for T in eltypes
    A = randn(T, (3, 5, 4, 6))
    p = ((3, 1), (4, 2))
    for conjA in (false, true), (α, β) in ((1, 0), (randn(T), 0), (randn(T), randn(T)))
        C = randn(T, (4, 3, 6, 5))
        Cref = copy(C)
        iszero(β) && (poison!(C); poison!(Cref))
        @test tensoradd!(C, A, p, conjA, α, β, tblis) ≈
            tensoradd!(Cref, A, p, conjA, α, β, reference)
    end

    # non-contiguous input and output through views
    Aview = view(randn(T, (6, 10, 8, 12)), 1:2:6, 1:2:10, 1:2:8, 1:2:12)
    C = randn(T, (4, 3, 6, 5))
    Cref = copy(C)
    @test tensoradd!(C, Aview, p, true, 2, 1, tblis) ≈
        tensoradd!(Cref, Aview, p, true, 2, 1, reference)

    # an `Adjoint` input already carries a conjugation, which has to combine with `conjA`
    Aadj = adjoint(randn(T, (5, 3)))
    for conjA in (false, true)
        C = randn(T, (5, 3))
        Cref = copy(C)
        @test tensoradd!(C, Aadj, ((2, 1), ()), conjA, 2, 1, tblis) ≈
            tensoradd!(Cref, Aadj, ((2, 1), ()), conjA, 2, 1, reference)
    end
end

@testset "tensortrace! (eltype = $T)" for T in eltypes
    A = randn(T, (5, 3, 4, 5, 3, 2))
    p = ((6, 3), ())
    q = ((1, 2), (4, 5))
    for conjA in (false, true), (α, β) in ((1, 0), (randn(T), randn(T)))
        C = randn(T, (2, 4))
        Cref = copy(C)
        iszero(β) && (poison!(C); poison!(Cref))
        @test tensortrace!(C, A, p, q, conjA, α, β, tblis) ≈
            tensortrace!(Cref, A, p, q, conjA, α, β, reference)
    end

    # trace all the way down to a scalar
    B = randn(T, (4, 4))
    C = fill(convert(T, NaN))
    Cref = fill(convert(T, NaN))
    @test tensortrace!(C, B, ((), ()), ((1,), (2,)), false, 1, 0, tblis)[] ≈ tr(B)
end

@testset "tensorcontract! (eltype = $T)" for T in eltypes
    # open indices of A are its dimensions 3, 1, 4 (sizes 5, 3, 3), contracted are 2 and 5
    # (sizes 20, 4); open indices of B are its dimensions 4 and 2 (sizes 3, 6)
    A = randn(T, (3, 20, 5, 3, 4))
    B = randn(T, (4, 6, 20, 3))
    pA = ((3, 1, 4), (2, 5))
    pB = ((3, 1), (4, 2))
    pAB = ((3, 1, 4), (5, 2))

    for conjA in (false, true), conjB in (false, true),
            (α, β) in ((1, 0), (randn(T), 0), (randn(T), randn(T)))

        C = randn(T, (3, 5, 3, 6, 3))
        Cref = copy(C)
        iszero(β) && (poison!(C); poison!(Cref))
        @test tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, tblis) ≈
            tensorcontract!(Cref, A, pA, conjA, B, pB, conjB, pAB, α, β, reference)
    end

    # outer product: no contracted indices at all
    @testset "outer product" begin
        A2 = randn(T, (3, 4))
        B2 = randn(T, (5,))
        pA2, pB2, pAB2 = ((1, 2), ()), ((), (1,)), ((3, 1), (2,))
        C = fill(convert(T, NaN), (5, 3, 4))
        Cref = copy(C)
        @test tensorcontract!(C, A2, pA2, true, B2, pB2, false, pAB2, 1, 0, tblis) ≈
            tensorcontract!(Cref, A2, pA2, true, B2, pB2, false, pAB2, 1, 0, reference)
    end

    # full contraction: zero-dimensional output
    @testset "full contraction" begin
        A2 = randn(T, (3, 4))
        B2 = randn(T, (4, 3))
        pA2, pB2, pAB2 = ((), (1, 2)), ((2, 1), ()), ((), ())
        C = fill(convert(T, NaN))
        Cref = fill(convert(T, NaN))
        @test tensorcontract!(C, A2, pA2, true, B2, pB2, true, pAB2, 1, 0, tblis)[] ≈
            tensorcontract!(Cref, A2, pA2, true, B2, pB2, true, pAB2, 1, 0, reference)[]
    end

    # non-contiguous factors
    @testset "strided views" begin
        Av = view(randn(T, (6, 8)), 1:2:6, 1:2:8)
        Bv = view(randn(T, (8, 10)), 1:2:8, 1:2:10)
        C = fill(convert(T, NaN), (3, 5))
        Cref = copy(C)
        pA2, pB2, pAB2 = ((1,), (2,)), ((1,), (2,)), ((1, 2), ())
        @test tensorcontract!(C, Av, pA2, false, Bv, pB2, true, pAB2, 1, 0, tblis) ≈
            tensorcontract!(Cref, Av, pA2, false, Bv, pB2, true, pAB2, 1, 0, reference)
    end
end

@testset "argument checking" begin
    A = randn(Float64, (4, 4))
    B = randn(Float64, (4, 4))
    # aliasing must be rejected rather than silently producing garbage
    @test_throws ArgumentError tensoradd!(A, A, ((2, 1), ()), false, 1, 0, tblis)
    @test_throws ArgumentError tensorcontract!(
        A, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1, 0, tblis
    )
    # shape errors are reported before anything reaches the library
    C = randn(Float64, (4, 3))
    @test_throws DimensionMismatch tensoradd!(C, A, ((1, 2), ()), false, 1, 0, tblis)
    @test_throws DimensionMismatch tensorcontract!(
        C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1, 2), ()), 1, 0, tblis
    )
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
    try
        A = randn(Float64, (40, 40, 20))
        B = randn(Float64, (20, 40, 40))
        Cref = similar(A, (40, 40, 40, 40))
        tensorcontract!(
            Cref, A, ((1, 2), (3,)), false, B, ((1,), (2, 3)), false,
            ((1, 2, 3, 4), ()), 1, 0, reference
        )
        for n in (1, 2)
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
