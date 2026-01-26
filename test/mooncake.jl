using TensorOperations
using TensorOperations: StridedBLAS, StridedNative, tensortrace!, Zero
using Test
using Mooncake
using Random

mode = Mooncake.ReverseMode
rng = Random.default_rng()
is_primitive = false

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
    @testset for α in (Zero(), rand(T)), β in (Zero(), rand(T))
        A = rand(T₁, (2, 3, 4, 2, 5))
        C = rand(T₂, size.(Ref(A), p[1]))

        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, false, α, β; atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, true, α, β; atol, rtol, mode, is_primitive)

        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, true, α, β, StridedBLAS(); atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, false, α, β, StridedNative(); atol, rtol, mode, is_primitive)
    end
end

@testset "tensoradd! ($T₁, $T₂)" for (T₁, T₂) in
    (
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
    @testset for α in (Zero(), rand(T)), β in (Zero(), rand(T))
        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, false, α, β; atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, true, α, β; atol, rtol, mode, is_primitive)

        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, false, α, β, StridedBLAS(); atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, true, α, β, StridedNative(); atol, rtol, mode, is_primitive)
    end
end

@testset "tensorcontract! ($T₁, $T₂)" for (T₁, T₂) in
    (
        (Float64, Float64),
        (Float32, Float64),
        (ComplexF64, ComplexF64),
        (Float64, ComplexF64),
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

    @testset for α in (Zero(), randn(T)), β in (Zero(), randn(T))
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, false, B, pB, false, pAB, α, β; atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, true, B, pB, false, pAB, α, β; atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, false, B, pB, true, pAB, α, β; atol, rtol, mode, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, true, B, pB, true, pAB, α, β; atol, rtol, mode, is_primitive)

        Mooncake.TestUtils.test_rule(
            rng,
            tensorcontract!, C, A, pA, false, B, pB, false, pAB, α, β, StridedBLAS();
            atol, rtol, mode, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng,
            tensorcontract!, C, A, pA, true, B, pB, false, pAB, α, β, StridedNative();
            atol, rtol, mode, is_primitive
        )
    end
end

@testset "tensorscalar ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    C = Array{T, 0}(undef, ())
    fill!(C, rand(T))
    Mooncake.TestUtils.test_rule(rng, tensorscalar, C; atol, rtol, mode, is_primitive)
end
