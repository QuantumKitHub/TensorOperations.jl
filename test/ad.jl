using TensorOperations
using TensorOperations: StridedBLAS, StridedNative
using Test
using ChainRulesTestUtils

ChainRulesTestUtils.test_method_tables()

@testset "tensortrace! ($T₁, $T₂)" for (T₁, T₂) in
                                       ((Float64, Float64), (Float32, Float64), (ComplexF64, ComplexF64), (Float64, ComplexF64))
    T = promote_type(T₁, T₂)
    atol = max(precision(T₁), precision(T₂))
    rtol = max(precision(T₁), precision(T₂))

    p = ((3, 5, 2), ())
    q = ((1,), (4,))
    α = rand(T)
    β = rand(T)
    A = rand(T₁, (2, 3, 4, 2, 5))
    C = rand(T₂, size.(Ref(A), p[1]))

    test_rrule(tensortrace!, C, A, p, q, false, α, β; atol, rtol)
    test_rrule(tensortrace!, C, A, p, q, true, α, β; atol, rtol)

    test_rrule(tensortrace!, C, A, p, q, true, α, β, StridedBLAS(); atol, rtol)
    test_rrule(tensortrace!, C, A, p, q, false, α, β, StridedNative(); atol, rtol)

    # Test double traces
    p2 = ((2,), ())
    q2 = ((1, 3), (4, 5)) # traces 1-4 and 3-5
    A2 = rand(T₁, (2, 3, 4, 2, 4))
    C2 = rand(T₂, size.(Ref(A2), p2[1]))

    test_rrule(tensortrace!, C2, A2, p2, q2, false, α, β; atol, rtol)
end

@testset "tensoradd! ($T₁, $T₂)" for (T₁, T₂) in
                                     ((Float64, Float64), (Float32, Float64), (ComplexF64, ComplexF64), (Float64, ComplexF64))
    T = promote_type(T₁, T₂)
    atol = max(precision(T₁), precision(T₂))
    rtol = max(precision(T₁), precision(T₂))

    pA = ((2, 1, 4, 3, 5), ())
    A = rand(T₁, (2, 3, 4, 2, 1))
    C = rand(T₂, size.(Ref(A), pA[1]))
    α = rand(T)
    β = rand(T)
    test_rrule(tensoradd!, C, A, pA, false, α, β; atol, rtol)
    test_rrule(tensoradd!, C, A, pA, true, α, β; atol, rtol)

    test_rrule(tensoradd!, C, A, pA, false, α, β, StridedBLAS(); atol, rtol)
    test_rrule(tensoradd!, C, A, pA, true, α, β, StridedNative(); atol, rtol)
end

@testset "tensorcontract! ($T₁, $T₂)" for (T₁, T₂) in
                                          ((Float64, Float64), (Float32, Float64), (ComplexF64, ComplexF64), (Float64, ComplexF64))
    T = promote_type(T₁, T₂)
    atol = max(precision(T₁), precision(T₂))
    rtol = max(precision(T₁), precision(T₂))

    pAB = ((3, 2, 4, 1), ())
    pA = ((2, 4, 5), (1, 3))
    pB = ((2, 1), (3,))

    A = rand(T₁, (2, 3, 4, 2, 5))
    B = rand(T₂, (4, 2, 3))
    C = rand(T, (5, 2, 3, 3))
    α = randn(T)
    β = randn(T)

    test_rrule(tensorcontract!, C, A, pA, false, B, pB, false, pAB, α, β; atol, rtol)
    test_rrule(tensorcontract!, C, A, pA, true, B, pB, false, pAB, α, β; atol, rtol)
    test_rrule(tensorcontract!, C, A, pA, false, B, pB, true, pAB, α, β; atol, rtol)
    test_rrule(tensorcontract!, C, A, pA, true, B, pB, true, pAB, α, β; atol, rtol)

    test_rrule(
        tensorcontract!, C, A, pA, false, B, pB, false, pAB, α, β, StridedBLAS();
        atol, rtol
    )
    test_rrule(
        tensorcontract!, C, A, pA, true, B, pB, false, pAB, α, β, StridedNative();
        atol, rtol
    )
end

@testset "tensorscalar ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    C = Array{T,0}(undef, ())
    fill!(C, rand(T))
    test_rrule(tensorscalar, C; atol, rtol)
end
