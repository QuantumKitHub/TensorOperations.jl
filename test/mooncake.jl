using TensorOperations
using TensorOperations: StridedBLAS, StridedNative, tensortrace!, tensoralloc, Zero
using Test
using Mooncake
using Random

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

        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, false, α, β; atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, true, α, β; atol, rtol, is_primitive)

        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, true, α, β, StridedBLAS(); atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensortrace!, C, A, p, q, false, α, β, StridedNative(); atol, rtol, is_primitive)
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
        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, false, α, β; atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, true, α, β; atol, rtol, is_primitive)

        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, false, α, β, StridedBLAS(); atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensoradd!, C, A, pA, true, α, β, StridedNative(); atol, rtol, is_primitive)
    end
end

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

    @testset for α in (Zero(), randn(T)), β in (Zero(), randn(T))
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, false, B, pB, false, pAB, α, β; atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, true, B, pB, false, pAB, α, β; atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, false, B, pB, true, pAB, α, β; atol, rtol, is_primitive)
        Mooncake.TestUtils.test_rule(rng, tensorcontract!, C, A, pA, true, B, pB, true, pAB, α, β; atol, rtol, is_primitive)

        Mooncake.TestUtils.test_rule(
            rng,
            tensorcontract!, C, A, pA, false, B, pB, false, pAB, α, β, StridedBLAS();
            atol, rtol, is_primitive
        )
        Mooncake.TestUtils.test_rule(
            rng,
            tensorcontract!, C, A, pA, true, B, pB, false, pAB, α, β, StridedNative();
            atol, rtol, is_primitive
        )
    end
end

@testset "tensorscalar ($T)" for T in (Float32, Float64, ComplexF64)
    atol = precision(T)
    rtol = precision(T)

    C = Array{T, 0}(undef, ())
    fill!(C, rand(T))
    Mooncake.TestUtils.test_rule(rng, tensorscalar, C; atol, rtol, is_primitive)
end

# the intermediate `A * B` of the first block is needed in the pullback after the second block
# has reused (`BufferAllocator`) or the first block has freed (`ManualAllocator`) its storage
function mooncake_two_tensor_blocks(A, B, D, allocator)
    @tensor allocator = allocator C[a, b] := A[a, c] * B[c, d] * D[d, b]
    @tensor allocator = allocator E[a, b] := D[a, c] * B[c, d] * C[d, b]
    return sum(abs2, E)
end

@testset "temporaries of $(nameof(typeof(allocator)))" for allocator in
    (TensorOperations.ManualAllocator(), TensorOperations.BufferAllocator())
    A, B, D = randn(rng, 4, 4), randn(rng, 4, 4), randn(rng, 4, 4)
    function gradient(allocator)
        f = (A, B, D) -> mooncake_two_tensor_blocks(A, B, D, allocator)
        cache = Mooncake.prepare_gradient_cache(f, A, B, D)
        val, (_, dA, dB, dD) = Mooncake.value_and_gradient!!(cache, f, A, B, D)
        return val, copy(dA), copy(dB), copy(dD)
    end
    ref = gradient(TensorOperations.DefaultAllocator())
    res = gradient(allocator)
    @test all(res .≈ ref)
    # repeated calls reuse the prepared rules without leaking or reusing freed temporaries
    @test all(gradient(allocator) .≈ ref)
end

@testset "No double free for ManualAllocator" begin
    allocator = TensorOperations.ManualAllocator()
    C_dC, pb = Mooncake.rrule!!(
        Mooncake.zero_fcodual(tensoralloc), Mooncake.zero_fcodual(Matrix{Float64}),
        Mooncake.zero_fcodual((16, 16)), Mooncake.zero_fcodual(Val(true)), Mooncake.zero_fcodual(allocator)
    )
    @test Mooncake.primal(C_dC) isa TensorOperations.PtrArrays.PtrArray{Float64, 2}
    @test pb(Mooncake.NoRData()) == ntuple(Returns(Mooncake.NoRData()), 5)
    @test pb(Mooncake.NoRData()) == ntuple(Returns(Mooncake.NoRData()), 5)
end
