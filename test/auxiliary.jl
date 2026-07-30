@testset "macro methods" begin
    @testset "tensorexpressions" begin
        using TensorOperations: isassignment, isdefinition, getlhs, getrhs, isindex,
            istensor, isgeneraltensor, istensorexpr, isscalarexpr,
            hastraceindices,
            hastraceindices, getindices, getallindices,
            normalizeindex, instantiate_scalar,
            instantiate_scalartype,
            decomposetensor, decomposegeneraltensor
        using VectorInterface: One

        @test isassignment(:(a[-1, -2, -3] = b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test isassignment(:(a[-1, -2, -3] += b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test isassignment(:(a[-1, -2, -3] -= b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test !isassignment(:(a[-1, -2, -3] := b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test !isassignment(:(a[-1, -2, -3] ≔ b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test !isassignment(:(b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))

        @test !isdefinition(:(a[-1, -2, -3] = b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test !isdefinition(:(a[-1, -2, -3] += b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test !isdefinition(:(a[-1, -2, -3] -= b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test isdefinition(:(a[-1, -2, -3] := b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test isdefinition(:(a[-1, -2, -3] ≔ b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))
        @test !isdefinition(:(b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1]))

        lhs = :(a[-1, -2, -3])
        rhs = :(b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1])
        for (getside, side) in ((getlhs, lhs), (getrhs, rhs))
            @test getside(:(a[-1, -2, -3] = b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1])) ==
                side
            @test getside(:(a[-1, -2, -3] += b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1])) ==
                side
            @test getside(:(a[-1, -2, -3] -= b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1])) ==
                side
            @test getside(:(a[-1, -2, -3] := b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1])) ==
                side
            @test getside(:(a[-1, -2, -3] ≔ b[-1, -2, 1] * c[1, -3] + d[-2, -3, -1])) ==
                side
        end

        @test isindex(:a)
        @test normalizeindex(:a) == :a
        @test isindex(:(a'))
        @test normalizeindex(:(a')) == :(a′)
        @test isindex(:(a'''))
        @test normalizeindex(:(a''')) == :(a′′′)
        @test isindex(:(β))
        @test normalizeindex(:(β)) == :(β)
        @test isindex(:(β'))
        @test normalizeindex(:(β')) == :(β′)
        @test isindex(:(3))
        @test normalizeindex(:(3)) == 3
        @test isindex(:(-5))
        @test normalizeindex(:(-5)) == -5
        @test !isindex(:('a'))
        @test !isindex(:(a + b))
        @test !isindex(:("x"))
        @test !isindex(:(5.1))

        @test istensor(:(a[1, 2, 3]))
        @test decomposetensor(:(a[1, 2, 3])) == (:a, Any[1, 2, 3], Any[])
        @test istensor(:(a[5][a b c]))
        @test decomposetensor(:(a[5][a b c])) == (:(a[5]), Any[:a, :b, :c], Any[])
        @test istensor(:(cos(y)[a b c; 1 2 3]))
        @test decomposetensor(:(cos(y)[a b c; 1 2 3])) ==
            (:(cos(y)), Any[:a, :b, :c], Any[1, 2, 3])
        @test istensor(:(x[(); (1, 2, 3)]))
        @test decomposetensor(:(x[(); (1, 2, 3)])) == (:x, Any[], Any[1, 2, 3])
        @test !istensor(:(2 * a[1, 2, 3]))
        @test !istensor(:(a[1 2 3; 4 5 6; 7 8 9]))
        @test !istensor(:(conj(a[5][a b c])))
        @test !istensor(:(cos(y)[a b c; 1 2 3] * b[4, 5]))
        @test !istensor(:(3 + 5))

        @test isgeneraltensor(:(conj(a[1, 2, 3])))
        @test decomposegeneraltensor(:(conj(a[1, 2, 3]))) ==
            (:a, Any[1, 2, 3], [], One(), true)
        @test isgeneraltensor(:(x * a[5][a b c]))
        @test decomposegeneraltensor(:(x * a[5][a b c])) ==
            (:(a[5]), Any[:a, :b, :c], [], :x, false)
        @test isgeneraltensor(:(x * x * a[5][a, b, c]))
        @test decomposegeneraltensor(:(x * x * a[5][a, b, c])) ==
            (:(a[5]), Any[:a, :b, :c], [], :(x * x), false)
        @test isgeneraltensor(:(x * a[5][a, b, c] * x))
        @test decomposegeneraltensor(:(x * a[5][a, b, c] * x)) ==
            (:(a[5]), Any[:a, :b, :c], [], :(x * x), false)
        @test isgeneraltensor(:(a[5][a, b, c] * x / y))
        @test decomposegeneraltensor(:(a[5][a, b, c] / y * x)) ==
            (:(a[5]), Any[:a, :b, :c], [], :((1 / y) * x), false)
        @test isgeneraltensor(:(x / y * a[5][a, b, c] * y / x))
        @test decomposegeneraltensor(:(x / y * a[5][a, b, c] / y * x)) ==
            (:(a[5]), Any[:a, :b, :c], [], :(((x / y) / y) * x), false)
        @test isgeneraltensor(:(3 * conj(a * cos(y)[a b c; 1 2 3])))
        @test decomposegeneraltensor(:(3 * conj(a * cos(y)[a b c; 1 2 3]))) ==
            (:(cos(y)), Any[:a, :b, :c], Any[1, 2, 3], :(3 * conj(a)), true)
        @test !isgeneraltensor(:(1 / a[1, 2, 3]))
        @test !isgeneraltensor(:(a[1 2 3; 4 5 6] \ x))
        @test !isgeneraltensor(:(cos(y)[a b c; 1 2 3] * b[4, 5]))
        @test !isgeneraltensor(:(x * y * cos(y)[a b c; 1 2 3] * b[4, 5]))
        @test !isgeneraltensor(:(x / y * cos(y)[a b c; 1 2 3] * b[4, 5] / z * v))
        @test !isgeneraltensor(:(3 + 5))

        @test hastraceindices(:(a[x, y, z, 1, x]))
        @test hastraceindices(:(a[x y z; 1 x]))
        @test !hastraceindices(:(a[-1, y, z, 1, x]))
        @test !hastraceindices(:(a[-1 y z; 1 x]))

        @test isscalarexpr(:a)
        @test isscalarexpr(2.0)
        @test isscalarexpr(:(2.0 + im * 3))
        @test isscalarexpr(:(3 * a + c))
        @test isscalarexpr(:(sin(x) + exp(-y)))
        @test isscalarexpr(:(tensorscalar(a[x, y] * b[y, x])))
        @test !isscalarexpr(:(a[x, y] * b[y, x]))
        @test !isscalarexpr(:(3 * tensorscalar(a[x, y] * b[y, x]) + conj(c[z])))

        @test instantiate_scalartype(:(a[1, 2, 3] * b[3, 4, 5] + c[1, 2, 4, 5])) ==
            :(promote_add(promote_contract(scalartype(a), scalartype(b)), scalartype(c)))
    end

    @testset "index verification" begin
        using TensorOperations: verifyindices, normalizeindices

        # every label appears once or exactly twice within a single term; different terms of
        # a sum, different statements and explicit `tensorscalar` calls are separate scopes
        valid = (
            :(a[-1, -2] := b[-1, 1] * c[1, -2]),
            :(a[-1, -2] := b[-1, 1] * c[1, -2] + d[-1, 1] * e[1, -2]),
            :(a[-1, -2] := b[-1, 1] * (c[1, 2] + d[1, 2]) * e[2, -2]),
            :(a[-1, -2] := b[-1, 1] * c[1, -2] * (d[2, 2])),
            :(a[x, y] := b[x, y] - 2 * c[y, x]),
            :(a[-1] := b[-1, 1, 1]),
            :(a[x, y] := b[x, z] * tensorscalar(c[z, z]) * d[z, y]),
            :(a[x, y] := b[x′, x] * c[x′, y]),
            quote
                a[-1, -2] := b[-1, 1] * c[1, -2]
                d[-1, -2] := e[-1, 1] * f[1, -2]
            end,
        )
        for ex in valid
            @test verifyindices(normalizeindices(ex)) isa Expr
        end

        # a label consumed by a trace or contraction cannot be reused within the same term,
        # and all terms of a sum should have matching open indices
        invalid = (
            :(a[-1, -2, -3; -4, -5] := b[-2, 1] * c[1, -3; 1, -5] * d[-1, 1; -4]), # issue #288
            :(a[-1, -2] := b[-1, 1; 1, -2] * c[1, 1]),
            :(a[-1, -2] := b[-1, 1] * c[1, -2] * d[1, 1, -3, -3]),
            :(a[-1, -2] := b[-1, 1] * c[1, -2] * (d[1, 1])),
            # parentheses do not introduce a new scope for the labels
            :(a[-1, -2] := (b[-1, 1] * c[1, -3]) * (d[-3, 1] * e[1, -2])),
            :(a[-1, -2] := b[-1, 1] * c[1, -3] * d[-3, 1] * e[1, -2]),
            :(a[-1] := b[1, 1, 1]),
            :(a[-1, -2] := b[-1, -2] + c[-1, 1]),
            :(a[x, y] := b[x′, x] * c[x′, y] * d[x', 1]), # primes are equal after normalizing
        )
        for ex in invalid
            @test_throws ArgumentError verifyindices(normalizeindices(ex))
        end

        # errors are thrown at macro expansion time
        @test_throws ArgumentError @macroexpand(
            @tensor T[-1, -2, -3; -4, -5] := A[-2, 1] * B[1, -3; 1, -5] * C[-1, 1; -4]
        )
        @test_throws ArgumentError @macroexpand(
            @tensoropt T[-1, -2] := A[-1, 1] * B[1, -2] * C[1, 1]
        )
    end

    @testset "parsecost" begin
        using TensorOperations: parsecost, Power
        @test parsecost(:(3 / 5)) === 3 / 5
        @test parsecost(:(5 + 2)) === 5 + 2
        @test parsecost(:(Int128(2 * 8))) === Int128(2 * 8)
        @test parsecost(:(float(2 * 8))) === float(2 * 8)
        @test parsecost(:(big(2 * 8))) isa BigInt
        @test parsecost(:(x)) == Power{:x}(1, 1)
        @test parsecost(:(3 * x^3)) == Power{:x}(3, 3)
        @test parsecost(:(3 * x^3 + 2)) == Power{:x}(3, 3) + 2
        @test parsecost(:((3 * x^3 - 2) / 5)) == (Power{:x}(3, 3) - 2) / 5
    end
end

@testset "methods for tensoropt" begin
    @testset "set methods" for T in (UInt32, UInt64, UInt128, BitSet, BitVector)
        maxint = 32
        A = Set(rand(1:maxint, 20))
        B = Set(rand(1:maxint, 20))
        U = union(A, B)
        I = intersect(A, B)
        S = setdiff(A, B)

        A2 = TensorOperations.storeset(T, A, maxint)
        B2 = TensorOperations.storeset(T, B, maxint)
        U2 = TensorOperations.storeset(T, U, maxint)
        I2 = TensorOperations.storeset(T, I, maxint)
        S2 = TensorOperations.storeset(T, S, maxint)

        @test TensorOperations._union(A2, B2) == U2
        @test TensorOperations._intersect(A2, B2) == I2
        @test TensorOperations._setdiff(A2, B2) == S2
        @test TensorOperations._isemptyset(TensorOperations.storeset(T, [], maxint))
        @test !TensorOperations._isemptyset(TensorOperations.storeset(T, [1], maxint))
    end

    @testset "poly" begin
        using TensorOperations: Power, degree, Poly
        x = Power{:x}(1, 1)
        x⁰ = Power{:x}(1, 0)
        x⁵ = Power{:x}(1, 5)
        @test x^5 == x⁵ == x * x * x * x * x
        @test x^0 == x⁰ == one(x)
        @test zero(x) == 0 * x
        @test (5 * x) / 5 == x
        @test degree(x) == 1
        @test x + x^3 + x^5 isa Poly
        p = x - x^3 + x^5
        coeffs = [0, 1, 0, -1, 0, 1]
        @test p == Poly{:x}(coeffs)
        @test 2.5 * p == Poly{:x}(2.5 * coeffs) == p * 2.5
        @test p / 2 == Poly{:x}(coeffs / 2) == 2 \ p
        @test p * p == x^10 + x^6 + x^2 - 2 * x^8 + 2 * x^6 - 2 * x^4 ==
            (x^10 + x^6 + x^2 + 1) - 2 * (x^8 - x^6 + x^4) - 1
        p = x + x^3 + x^5
        for i in 0:6
            @test p[i] == isodd(i)
        end
        @test p + p / 2 == 1.5 * p
        @test p * one(p) == p == p * one(x)
        @test p * zero(p) == 0 == p * zero(x)
        @test p <= p * x
        @test x * p == p * x
    end

    @testset "expandconj" begin
        @test TensorOperations.expandconj(:(conj(a[x, y, z] * b[x, y, z]))) ==
            :(conj(a[x, y, z]) * conj(b[x, y, z]))
        @test TensorOperations.expandconj(
            :(conj(a[x, y, z] * 2 * c[x, y, z] + 12 + A[1, 2] * B[3, 4] \ 1))
        ) ==
            :(
            conj(a[x, y, z]) * conj(2) * conj(c[x, y, z]) + conj(12) +
                conj(A[1, 2]) * conj(B[3, 4]) \ conj(1)
        )
    end

    @testset "cost methods" begin end
end

@testset "methods for indices" begin
    using TensorOperations: add_indices, add_labels, contract_indices, contract_labels,
        trace_indices, trace_labels
    using TensorOperations: IndexError

    @testset "add" begin
        @test add_indices((:a, :b, :c), (:a, :b, :c)) == ((1, 2, 3), ())
        @test add_indices((:a, :b, :c), (:c, :b, :a)) == ((3, 2, 1), ())
        @test add_indices((1, :b, 3), (3, 1, :b)) == ((3, 1, 2), ())

        @test_throws IndexError add_indices((:a, :b, :c), (:c, :a, :b, :d))
        @test_throws IndexError add_indices((:a, :b, :c), (:a, :b, :d))

        # only tests for all indices to the left, labels defaults to this
        for pA in (((1, 2, 3), ()), ((1, 3, 2, 4), ()), ((3, 2, 1, 4), ()))
            @test add_indices(add_labels(pA)...) == pA
        end
    end

    @testset "trace" begin
        @test trace_indices((:a, :b, :c, :d, :d), (:a, :b, :c)) ==
            (((1, 2, 3), ()), ((4,), (5,)))
        @test trace_indices((:a, :b, :c, :d, :d), (:c, :b, :a)) ==
            (((3, 2, 1), ()), ((4,), (5,)))
        @test trace_indices((4, :b, 3, 1, 4), (3, 1, :b)) ==
            (((3, 4, 2), ()), ((1,), (5,)))

        @test_throws IndexError trace_indices((:a, :b, :c, :d, :d), (:a, :b, :d))
        @test_throws IndexError trace_indices((:a, :b, :c, :d, :d), (:a, :b, :c, :d, :d))

        # only tests for all indices to the left, labels defaults to this
        for (p, q) in
            (
                (((1, 2, 3), ()), ((4,), (5,))), (((3, 4, 2), ()), ((1,), (5,))),
                (((3, 1, 4, 5), ()), ((6, 7), (2, 8))),
            )
            p′, q′ = trace_indices(trace_labels(p, q)...)
            @test p′ == p
            @test issetequal(
                [Set([i1, i2]) for (i1, i2) in zip(q...)],
                [Set([i1, i2]) for (i1, i2) in zip(q′...)]
            )
        end
    end

    @testset "contract" begin
        @test contract_indices((:a, :b, :c), (:c, :d), (:a, :b, :d)) ==
            (((1, 2), (3,)), ((1,), (2,)), ((1, 2, 3), ()))
        @test contract_indices((:c, :a, :b), (:c, :d), (:a, :b, :d)) ==
            (((2, 3), (1,)), ((1,), (2,)), ((1, 2, 3), ()))

        @test_throws IndexError contract_indices((:a, :b, :c), (:c, :d), (:a, :b, :d, :e))
        @test_throws IndexError contract_indices((:a, :b, :c), (:c, :d), (:a, :b, :e))

        for (pA, pB, pAB) in
            (
                (((1, 2), (3,)), ((1,), (2,)), ((1, 2, 3), ())),
                (((2, 3), (1,)), ((1,), (2,)), ((1, 2, 3), ())),
                (((1, 3), (2, 4)), ((1, 3), (2,)), ((2, 3, 1), ())),
            )
            @test contract_indices(contract_labels(pA, pB, pAB)...) == (pA, pB, pAB)
        end
    end
end
