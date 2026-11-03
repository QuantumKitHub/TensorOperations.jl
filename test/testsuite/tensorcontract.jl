# sizes of A and B, and the permutations pA, pB and pAB
const CONTRACT_CASES = [
    ((3, 6, 5, 3, 4), ((3, 1, 4), (2, 5)), (4, 2, 6, 3), ((3, 1), (2, 4)), ((3, 1, 4), (5, 2))),
    ((4, 5), ((1,), (2,)), (5, 3), ((1,), (2,)), ((1, 2), ())), # matrix-matrix
    ((5, 4), ((2,), (1,)), (3, 5), ((2,), (1,)), ((2, 1), ())), # transposed matrix-matrix
    ((4, 5), ((1,), (2,)), (5,), ((1,), ()), ((1,), ())), # matrix-vector
    ((5,), ((), (1,)), (5,), ((1,), ()), ((), ())), # inner product
    ((3, 4), ((), (1, 2)), (4, 3), ((2, 1), ()), ((), ())), # full contraction
    ((3, 4), ((1, 2), ()), (5,), ((), (1,)), ((3, 1), (2,))), # outer product
    ((), ((), ()), (), ((), ()), ((), ())), # zero-dimensional
    ((), ((), ()), (3, 4), ((), (1, 2)), ((2, 1), ())), # zero-dimensional times matrix
    ((1, 3, 1), ((1, 3), (2,)), (3, 1), ((1,), (2,)), ((1, 2, 3), ())), # dimensions of size one
    # index layouts that cannot be fused into matrix multiplications without permuting
    ((2, 3, 2, 3, 2), ((5, 1, 3), (4, 2)), (3, 2, 2, 3), ((1, 4), (2, 3)), ((2, 4, 1), (5, 3))),
]
const CONTRACT_ZERO_SIZE_CASES = [
    ((3, 0), ((1,), (2,)), (0, 4), ((1,), (2,)), ((1, 2), ())), # contracted dimension of size zero
    ((0, 3), ((1,), (2,)), (3, 4), ((1,), (2,)), ((1, 2), ())),
]

function contract_size(szA, pA, szB, pB, pAB)
    open = (map(i -> szA[i], pA[1])..., map(i -> szB[i], pB[2])...)
    return map(i -> open[i], linearize(pAB))
end

function _contract!(pA, conjA, pB, conjB, pAB, α, β)
    return (C, A, B, backend, allocator) -> tensorcontract!(
        C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, allocator
    )
end

@testsuite "tensorcontract!" (backend, options) -> begin
    cases = options.zero_size ? [CONTRACT_CASES; CONTRACT_ZERO_SIZE_CASES] : CONTRACT_CASES
    @testset "$T" for T in options.eltypes
        rtol = precision(T)
        @testset "sizes $szA, $szB, pA = $pA, pB = $pB, pAB = $pAB" for (szA, pA, szB, pB, pAB) in cases
            for conjA in (false, true), conjB in (false, true), (α, β) in scalars(T)
                A = randarray(T, szA)
                B = randarray(T, szB)
                C = maybe_poison!(randarray(T, contract_size(szA, pA, szB, pB, pAB)), β)
                f! = _contract!(pA, conjA, pB, conjB, pAB, α, β)
                check_inplace(f!, backend, options, C, A, B; rtol)
            end
        end

        szA, pA, szB, pB, pAB = first(CONTRACT_CASES)
        szC = contract_size(szA, pA, szB, pB, pAB)
        α, β = randn(rng, T), randn(rng, T)
        if options.views
            @testset "strided views" for conjA in (false, true), β′ in (β, Zero())
                f! = _contract!(pA, conjA, pB, !conjA, pAB, α, β′)
                for (A, B, C) in (
                        (stridedview(T, szA), randarray(T, szB), randarray(T, szC)),
                        (randarray(T, szA), stridedview(T, szB), randarray(T, szC)),
                        (randarray(T, szA), randarray(T, szB), stridedview(T, szC)),
                        (stridedview(T, szA), stridedview(T, szB), stridedview(T, szC)),
                    )
                    maybe_poison!(C, β′)
                    check_inplace(f!, backend, options, C, A, B; rtol)
                end
            end
        end
        if options.views && options.negative_strides
            @testset "negative strides" for conjA in (false, true)
                A = stridedview(T, szA; reversed = true)
                B = stridedview(T, szB; reversed = true)
                C = stridedview(T, szC; reversed = true)
                f! = _contract!(pA, conjA, pB, !conjA, pAB, α, β)
                check_inplace(f!, backend, options, C, A, B; rtol)
            end
        end
        if options.wrappers
            @testset "$wrapperA and $wrapperB inputs" for wrapperA in (adjoint, transpose),
                    wrapperB in (adjoint, transpose)

                for conjA in (false, true), conjB in (false, true)
                    A = wrapperA(randarray(T, (5, 4)))
                    B = wrapperB(randarray(T, (3, 5)))
                    C = randarray(T, (4, 3))
                    f! = _contract!(((1,), (2,)), conjA, ((1,), (2,)), conjB, ((1, 2), ()), α, β)
                    check_inplace(f!, backend, options, C, A, B; rtol)
                end
            end
        end
        if options.diagonal
            @testset "Diagonal" for conjA in (false, true), β′ in (β, Zero())
                n = 4
                D, D2 = Diagonal(randarray(T, (n,))), Diagonal(randarray(T, (n,)))
                A = randarray(T, (n, n, n))
                # C[i,j,k] = A[i,j,a] * D[a,k]
                C = maybe_poison!(randarray(T, (n, n, n)), β′)
                f! = _contract!(((1, 2), (3,)), conjA, ((1,), (2,)), !conjA, ((1, 2, 3), ()), α, β′)
                check_inplace(f!, backend, options, C, A, D; rtol)
                # C[k,i,j] = D[k,a] * A[i,a,j]
                C = maybe_poison!(randarray(T, (n, n, n)), β′)
                f! = _contract!(((1,), (2,)), conjA, ((2,), (1, 3)), !conjA, ((1, 2, 3), ()), α, β′)
                check_inplace(f!, backend, options, C, D, A; rtol)
                # C[i,j] = A[i,j,a] * D[a,a]: contracting both indices of the diagonal
                A4 = randarray(T, (n, n, n, n))
                C = maybe_poison!(randarray(T, (n, n)), β′)
                f! = _contract!(((1, 2), (3, 4)), conjA, ((1, 2), ()), !conjA, ((1, 2), ()), α, β′)
                check_inplace(f!, backend, options, C, A4, D; rtol)
                # C[a,c,b,d] = M[a,b] * D[c,d]: outer product
                M = randarray(T, (n, n))
                C = maybe_poison!(randarray(T, (n, n, n, n)), β′)
                f! = _contract!(((1, 2), ()), conjA, ((), (1, 2)), !conjA, ((1, 3, 2, 4), ()), α, β′)
                check_inplace(f!, backend, options, C, M, D; rtol)
                # diagonal times diagonal, into a dense matrix and into a diagonal
                f! = _contract!(((1,), (2,)), conjA, ((1,), (2,)), !conjA, ((1, 2), ()), α, β′)
                C = maybe_poison!(randarray(T, (n, n)), β′)
                check_inplace(f!, backend, options, C, D, D2; rtol)
                C = Diagonal(maybe_poison!(randarray(T, (n,)), β′))
                check_inplace(f!, backend, options, C, D, D2; rtol)
            end
        end
    end

    @testset "mixed eltypes C::$TC, A::$TA, B::$TB" for (TC, TA, TB) in options.mixed_eltypes
        szA, pA, szB, pB, pAB = first(CONTRACT_CASES)
        rtol = precision(TC, TA, TB)
        for conjA in (false, true), (α, β) in scalars(real(TC))
            A = randarray(TA, szA)
            B = randarray(TB, szB)
            C = maybe_poison!(randarray(TC, contract_size(szA, pA, szB, pB, pAB)), β)
            f! = _contract!(pA, conjA, pB, !conjA, pAB, α, β)
            check_inplace(f!, backend, options, C, A, B; rtol)
        end
    end

    @testset "allocating" for T in options.eltypes
        rtol = precision(T)
        # general contraction and full contraction to a zero-dimensional array
        for (szA, pA, szB, pB, pAB) in CONTRACT_CASES[[1, 6]], conjA in (false, true)
            A = randarray(T, szA)
            B = randarray(T, szB)
            α = randn(rng, T)
            f = (A, B, backend, allocator) -> tensorcontract(
                A, pA, conjA, B, pB, !conjA, pAB, α, backend, allocator
            )
            check_allocating(f, backend, options, A, B; rtol)
        end
        # the value of a full contraction is obtained with `tensorscalar`
        szA, pA, szB, pB, pAB = CONTRACT_CASES[6]
        A = randarray(T, szA)
        B = randarray(T, szB)
        Cref = tensorcontract(A, pA, false, B, pB, false, pAB, One(), options.reference)
        Adev, Bdev = todevice(options.arraytype, A), todevice(options.arraytype, B)
        C = tensorcontract(Adev, pA, false, Bdev, pB, false, pAB, One(), backend, options.allocator)
        @test tensorscalar(C) ≈ tensorscalar(Cref) rtol = rtol
    end

    @testset "argument checking" begin
        T = first(options.eltypes)
        A = todevice(options.arraytype, randarray(T, (3, 4, 5)))
        B = todevice(options.arraytype, randarray(T, (5, 4, 2)))
        C = todevice(options.arraytype, randarray(T, (3, 2)))
        allocator = options.allocator
        pA, pB, pAB = ((1,), (2, 3)), ((2, 1), (3,)), ((1, 2), ())
        @test tensorcontract!(C, A, pA, false, B, pB, false, pAB, 1, 0, backend, allocator) === C
        # wrong number of indices
        @test_throws IndexError tensorcontract!(C, A, ((1,), (2,)), false, B, pB, false, pAB, 1, 0, backend, allocator)
        @test_throws IndexError tensorcontract!(C, A, pA, false, B, ((2, 1), ()), false, pAB, 1, 0, backend, allocator)
        @test_throws IndexError tensorcontract!(C, A, pA, false, B, pB, false, ((1,), ()), 1, 0, backend, allocator)
        # different numbers of contracted indices
        @test_throws IndexError tensorcontract!(C, A, pA, false, B, ((2,), (1, 3)), false, ((1, 2, 3), ()), 1, 0, backend, allocator)
        # contracted dimensions of different size
        @test_throws DimensionMismatch tensorcontract!(C, A, pA, false, B, ((1, 2), (3,)), false, pAB, 1, 0, backend, allocator)
        # wrong output size
        @test_throws DimensionMismatch tensorcontract!(C, A, pA, false, B, pB, false, ((2, 1), ()), 1, 0, backend, allocator)
        if options.aliasing
            M = todevice(options.arraytype, randarray(T, (4, 4)))
            N = todevice(options.arraytype, randarray(T, (4, 4)))
            p = ((1,), (2,))
            @test_throws ArgumentError tensorcontract!(M, M, p, false, N, p, false, ((1, 2), ()), 1, 0, backend, allocator)
        end
    end
end
