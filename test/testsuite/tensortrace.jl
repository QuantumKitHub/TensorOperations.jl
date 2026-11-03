# sizes of A, the permutation p of the open indices and the pairs q of traced indices
const TRACE_CASES = [
    ((5, 3, 4, 5, 3, 2), ((6, 3), ()), ((1, 2), (4, 5))),
    ((2, 3, 4, 3), ((1,), (3,)), ((2,), (4,))),
    ((4, 4), ((), ()), ((1,), (2,))), # full trace
    ((3, 2, 3, 2), ((), ()), ((1, 2), (3, 4))), # full trace over two pairs
    ((3, 4), ((2, 1), ()), ((), ())), # nothing to trace
    ((3, 1, 2, 3, 1), ((3,), ()), ((1, 2), (4, 5))), # dimensions of size one
]
const TRACE_ZERO_SIZE_CASES = [
    ((0, 3, 0), ((2,), ()), ((1,), (3,))), # traced dimensions of size zero
    ((2, 0, 2), ((2,), ()), ((1,), (3,))),
]

trace_size(szA, p) = map(i -> szA[i], linearize(p))

function _trace!(p, q, conjA, α, β)
    return (C, A, backend, allocator) -> tensortrace!(C, A, p, q, conjA, α, β, backend, allocator)
end

@testsuite "tensortrace!" (backend, options) -> begin
    cases = options.zero_size ? [TRACE_CASES; TRACE_ZERO_SIZE_CASES] : TRACE_CASES
    @testset "$T" for T in options.eltypes
        rtol = precision(T)
        @testset "size $szA, p = $p, q = $q" for (szA, p, q) in cases
            for conjA in (false, true), (α, β) in scalars(T)
                A = randarray(T, szA)
                C = maybe_poison!(randarray(T, trace_size(szA, p)), β)
                check_inplace(_trace!(p, q, conjA, α, β), backend, options, C, A; rtol)
            end
        end

        szA, p, q = (5, 3, 4, 5, 3, 2), ((6, 3), ()), ((1, 2), (4, 5))
        α, β = randn(rng, T), randn(rng, T)
        szC = trace_size(szA, p)
        if options.views
            @testset "strided views" for conjA in (false, true), β′ in (β, Zero())
                for (A, C) in (
                        (stridedview(T, szA), randarray(T, szC)),
                        (randarray(T, szA), stridedview(T, szC)),
                        (stridedview(T, szA), stridedview(T, szC)),
                    )
                    maybe_poison!(C, β′)
                    check_inplace(_trace!(p, q, conjA, α, β′), backend, options, C, A; rtol)
                end
            end
        end
        if options.views && options.negative_strides
            @testset "negative strides" for conjA in (false, true)
                A = stridedview(T, szA; reversed = true)
                C = stridedview(T, szC; reversed = true)
                check_inplace(_trace!(p, q, conjA, α, β), backend, options, C, A; rtol)
            end
        end
        if options.wrappers
            @testset "$wrapper input" for wrapper in (adjoint, transpose), conjA in (false, true)
                A = wrapper(randarray(T, (4, 4)))
                C = randarray(T, ())
                check_inplace(_trace!(((), ()), ((1,), (2,)), conjA, α, β), backend, options, C, A; rtol)
            end
        end
    end

    # only the element types of the output and the first input
    mixed_eltypes = unique(map(Ts -> Ts[1:2], options.mixed_eltypes))
    @testset "mixed eltypes C::$TC, A::$TA" for (TC, TA) in mixed_eltypes
        szA, p, q = (5, 3, 4, 5, 3, 2), ((6, 3), ()), ((1, 2), (4, 5))
        rtol = precision(TC, TA)
        for conjA in (false, true), (α, β) in scalars(real(TC))
            A = randarray(TA, szA)
            C = maybe_poison!(randarray(TC, trace_size(szA, p)), β)
            check_inplace(_trace!(p, q, conjA, α, β), backend, options, C, A; rtol)
        end
    end

    @testset "allocating" for T in options.eltypes
        A = randarray(T, (5, 3, 4, 5, 3, 2))
        rtol = precision(T)
        for conjA in (false, true)
            α = randn(rng, T)
            f = (A, backend, allocator) -> tensortrace(
                A, ((6, 3), ()), ((1, 2), (4, 5)), conjA, α, backend, allocator
            )
            check_allocating(f, backend, options, A; rtol)
        end
    end

    @testset "argument checking" begin
        T = first(options.eltypes)
        A = todevice(options.arraytype, randarray(T, (5, 3, 4, 5, 3)))
        C = todevice(options.arraytype, randarray(T, (4,)))
        allocator = options.allocator
        # wrong number of indices
        @test_throws IndexError tensortrace!(C, A, ((3,), ()), ((1,), (4,)), false, 1, 0, backend, allocator)
        # index both open and traced
        @test_throws IndexError tensortrace!(C, A, ((3,), ()), ((1, 3), (4, 5)), false, 1, 0, backend, allocator)
        # traced dimensions of different size
        @test_throws DimensionMismatch tensortrace!(C, A, ((3,), ()), ((1, 2), (5, 4)), false, 1, 0, backend, allocator)
        # wrong output size
        C3 = todevice(options.arraytype, randarray(T, (3,)))
        @test_throws DimensionMismatch tensortrace!(C3, A, ((3,), ()), ((1, 2), (4, 5)), false, 1, 0, backend, allocator)
    end
end
