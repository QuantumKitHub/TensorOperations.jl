using TensorOperations, VectorInterface
using Enzyme, ChainRulesCore, EnzymeTestUtils

@testset "tensorcontract! ($T₁, $T₂)" for (T₁, T₂) in
    (
        (Float64, Float64),
        (Float32, Float64),
        (ComplexF64, ComplexF64),
        (Float64, ComplexF64),
        (ComplexF64, Float64),
    )
    T = promote_type(T₁, T₂)
    atol = max(precision(T₁), precision(T₂))
    rtol = max(precision(T₁), precision(T₂))

    pAB = ((3, 2, 4, 1), ())
    pA = ((2, 4, 5), (1, 3))
    pB = ((2, 1), (3,))

    A = rand(T₁, (2, 3, 4, 2, 5))
    B = rand(T₂, (4, 2, 3))
    C = rand(T, (5, 2, 3, 3))
    @testset for (α, β) in ((Zero(), Zero()), (randn(T), Zero()), (Zero(), randn(T)), (randn(T), randn(T)))
        Tα = α === Zero() ? Const : Active
        Tβ = β === Zero() ? Const : Active
        test_reverse(tensorcontract!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (false, Const), (B, Duplicated), (pB, Const), (false, Const), (pAB, Const), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(tensorcontract!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (false, Const), (B, Duplicated), (pB, Const), (true, Const), (pAB, Const), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(tensorcontract!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (true, Const), (B, Duplicated), (pB, Const), (true, Const), (pAB, Const), (α, Tα), (β, Tβ); atol, rtol)

        test_reverse(tensorcontract!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (false, Const), (B, Duplicated), (pB, Const), (false, Const), (pAB, Const), (α, Tα), (β, Tβ), (StridedBLAS(), Const); atol, rtol)
        test_reverse(tensorcontract!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (true, Const), (B, Duplicated), (pB, Const), (true, Const), (pAB, Const), (α, Tα), (β, Tβ), (StridedNative(), Const); atol, rtol)
    end
end

@testset "tensoradd! ($T₁, $T₂)" for (T₁, T₂) in (
        (Float64, Float64),
        (Float32, Float64),
        (ComplexF64, ComplexF64),
        (Float64, ComplexF64),
    )
    T = promote_type(T₁, T₂)
    atol = max(precision(T₁), precision(T₂))
    rtol = max(precision(T₁), precision(T₂))

    pA = ((2, 1, 4, 3, 5), ())
    A = rand(T₁, (2, 3, 4, 2, 1))
    C = rand(T₂, size.(Ref(A), pA[1]))
    @testset for (α, β) in ((Zero(), Zero()), (randn(T), Zero()), (Zero(), randn(T)), (randn(T), randn(T)))
        Tα = α === Zero() ? Const : Active
        Tβ = β === Zero() ? Const : Active
        test_reverse(tensoradd!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (false, Const), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(tensoradd!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (true, Const), (α, Tα), (β, Tβ); atol, rtol)

        test_reverse(tensoradd!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (false, Const), (α, Tα), (β, Tβ), (StridedBLAS(), Const); atol, rtol)
        test_reverse(tensoradd!, Duplicated, (C, Duplicated), (A, Duplicated), (pA, Const), (true, Const), (α, Tα), (β, Tβ), (StridedNative(), Const); atol, rtol)
    end
end

@testset "tensortrace! ($T₁, $T₂)" for (T₁, T₂) in
    (
        (Float64, Float64),
        (Float32, Float64),
        (ComplexF64, ComplexF64),
        (Float64, ComplexF64),
    )
    T = promote_type(T₁, T₂)
    atol = max(precision(T₁), precision(T₂))
    rtol = max(precision(T₁), precision(T₂))

    p = ((3, 5, 2), ())
    q = ((1,), (4,))
    A = rand(T₁, (2, 3, 4, 2, 5))
    C = rand(T₂, size.(Ref(A), p[1]))
    @testset for (α, β) in ((Zero(), Zero()), (randn(T), Zero()), (Zero(), randn(T)), (randn(T), randn(T)))
        Tα = α === Zero() ? Const : Active
        Tβ = β === Zero() ? Const : Active

        test_reverse(tensortrace!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (q, Const), (false, Const), (α, Tα), (β, Tβ); atol, rtol)
        test_reverse(tensortrace!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (q, Const), (true, Const), (α, Tα), (β, Tβ); atol, rtol)

        test_reverse(tensortrace!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (q, Const), (true, Const), (α, Tα), (β, Tβ), (StridedBLAS(), Const); atol, rtol)
        test_reverse(tensortrace!, Duplicated, (C, Duplicated), (A, Duplicated), (p, Const), (q, Const), (false, Const), (α, Tα), (β, Tβ), (StridedNative(), Const); atol, rtol)
    end
end

@testset "tensorscalar ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    C = Array{T, 0}(undef, ())
    fill!(C, rand(T))
    test_reverse(tensorscalar, Active, (C, Duplicated); atol, rtol)
end
