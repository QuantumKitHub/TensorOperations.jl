# sizes of A and the permutation pA
const ADD_CASES = [
    ((3, 4, 5), ((3, 1), (2,))),
    ((3, 4), ((1, 2), ())), # identity
    ((2, 3, 4), ((), (3, 2, 1))),
    ((), ((), ())), # zero-dimensional
    ((5,), ((1,), ())),
    ((1, 3, 1, 2), ((4, 2), (3, 1))), # dimensions of size one
    ((2, 1, 3, 2, 2, 3), ((5, 3, 1), (6, 2, 4))),
]
const ADD_ZERO_SIZE_CASES = [
    ((3, 0, 2), ((2, 3, 1), ())),
    ((0,), ((1,), ())),
]

add_size(szA, pA) = map(i -> szA[i], linearize(pA))

function _add!(pA, conjA, α, β)
    return (C, A, backend, allocator) -> tensoradd!(C, A, pA, conjA, α, β, backend, allocator)
end

@testsuite "tensoradd!" (backend, options) -> begin
    cases = options.zero_size ? [ADD_CASES; ADD_ZERO_SIZE_CASES] : ADD_CASES
    @testset "$T" for T in options.eltypes
        rtol = precision(T)
        @testset "size $szA, pA = $pA" for (szA, pA) in cases
            for conjA in (false, true), (α, β) in scalars(T)
                A = randarray(T, szA)
                C = maybe_poison!(randarray(T, add_size(szA, pA)), β)
                check_inplace(_add!(pA, conjA, α, β), backend, options, C, A; rtol)
            end
        end

        szA, pA = (3, 4, 5), ((3, 1), (2,))
        α, β = randn(rng, T), randn(rng, T)
        szC = add_size(szA, pA)
        if options.views
            @testset "strided views" for conjA in (false, true), β′ in (β, Zero())
                for (A, C) in (
                        (stridedview(T, szA), randarray(T, szC)),
                        (randarray(T, szA), stridedview(T, szC)),
                        (stridedview(T, szA), stridedview(T, szC)),
                    )
                    maybe_poison!(C, β′)
                    check_inplace(_add!(pA, conjA, α, β′), backend, options, C, A; rtol)
                end
            end
        end
        if options.views && options.negative_strides
            @testset "negative strides" for conjA in (false, true)
                A = stridedview(T, szA; reversed = true)
                C = stridedview(T, szC; reversed = true)
                check_inplace(_add!(pA, conjA, α, β), backend, options, C, A; rtol)
            end
        end
        if options.wrappers
            # `Adjoint` already carries a conjugation, which has to combine with `conjA`
            @testset "$wrapper input" for wrapper in (adjoint, transpose)
                for conjA in (false, true), pA in (((1, 2), ()), ((2,), (1,)))
                    A = wrapper(randarray(T, (4, 3)))
                    C = randarray(T, add_size(size(A), pA))
                    check_inplace(_add!(pA, conjA, α, β), backend, options, C, A; rtol)
                end
            end
        end
    end

    # only the element types of the output and the first input
    mixed_eltypes = unique(map(Ts -> Ts[1:2], options.mixed_eltypes))
    @testset "mixed eltypes C::$TC, A::$TA" for (TC, TA) in mixed_eltypes
        szA, pA = (3, 4, 5), ((3, 1), (2,))
        rtol = precision(TC, TA)
        for conjA in (false, true), (α, β) in scalars(real(TC))
            A = randarray(TA, szA)
            C = maybe_poison!(randarray(TC, add_size(szA, pA)), β)
            check_inplace(_add!(pA, conjA, α, β), backend, options, C, A; rtol)
        end
    end

    @testset "allocating" for T in options.eltypes
        A = randarray(T, (3, 4, 5))
        B = randarray(T, (5, 3, 4))
        rtol = precision(T)
        for conjA in (false, true), conjB in (false, true)
            α, β = randn(rng, T), randn(rng, T)
            f = (A, B, backend, allocator) -> tensoradd(
                A, ((1, 2, 3), ()), conjA, B, ((2, 3, 1), ()), conjB, α, β, backend, allocator
            )
            check_allocating(f, backend, options, A, B; rtol)
        end
    end

    @testset "argument checking" begin
        T = first(options.eltypes)
        A = todevice(options.arraytype, randarray(T, (3, 4, 5)))
        C = todevice(options.arraytype, randarray(T, (5, 3, 4)))
        allocator = options.allocator
        # wrong number of indices
        @test_throws IndexError tensoradd!(C, A, ((3, 1), ()), false, 1, 0, backend, allocator)
        # not a permutation
        @test_throws IndexError tensoradd!(C, A, ((3, 1, 1), ()), false, 1, 0, backend, allocator)
        # wrong sizes
        @test_throws DimensionMismatch tensoradd!(C, A, ((1, 2, 3), ()), false, 1, 0, backend, allocator)
        if options.aliasing
            B = todevice(options.arraytype, randarray(T, (4, 4)))
            @test_throws ArgumentError tensoradd!(B, B, ((2, 1), ()), false, 1, 0, backend, allocator)
        end
    end
end
