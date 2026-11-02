backendlist = (BaseCopy(), BaseView(), StridedNative(), StridedBLAS())

# test simple methods
#---------------------
@testset "simple methods with backend = $b" for b in backendlist
    @testset "tensorcopy" begin
        A = randn(Float64, (3, 5, 4, 6))
        p = (3, 1, 4, 2)
        C1 = permutedims(A, p)
        C2 = @inferred tensorcopy((p...,), A, (1:4...,); backend = b)
        C3 = @inferred tensorcopy(A, (p, ()), false, 1, b)
        @test C1 ≈ C2
        @test C2 == C3
        @test C1 ≈ ncon(Any[A], Any[[-2, -4, -1, -3]])
        @test_throws IndexError tensorcopy(1:4, A, 1:3)
        @test_throws IndexError tensorcopy(1:4, A, [1, 2, 2, 4])
    end

    @testset "tensoradd" begin
        A = randn(Float64, (3, 5, 4, 6))
        B = randn(Float64, (5, 6, 3, 4))
        p = (3, 1, 4, 2)
        C1 = A + permutedims(B, p)
        C2 = @inferred tensoradd(A, p, B, (1:4...,); backend = b)
        C3 = @inferred tensoradd(A, ((1:4...,), ()), false, B, (p, ()), false, 1.0, 1.0, b)
        @test C1 ≈ C2
        @test C2 == C3
        @test C1 ≈ A + ncon(Any[B], Any[[-2, -4, -1, -3]]; backend = b)
        @test_throws DimensionMismatch tensoradd(A, 1:4, B, 1:4)
    end

    @testset "tensortrace" begin
        A = randn(Float64, (50, 100, 100))
        C1 = zeros(50)
        for i in 1:50
            for j in 1:100
                C1[i] += A[i, j, j]
            end
        end
        C2 = tensortrace(A, [:a, :b, :b]; backend = b)
        C3 = ncon(Any[A], Any[[-1, 1, 1]]; backend = b)
        @test C1 ≈ C2
        @test C2 == C3
        A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
        C1 = zeros(4, 3, 3)
        for i1 in 1:4, i2 in 1:3, i3 in 1:3
            for j1 in 1:20, j2 in 1:5
                C1[i1, i2, i3] += A[i2, j1, j2, i3, j1, i1, j2]
            end
        end
        C2 = @inferred tensortrace((:e, :a, :d), A, (:a, :b, :c, :d, :b, :e, :c); backend = b)
        C3 = @inferred tensortrace(A, ((6, 1, 4), ()), ((2, 3), (5, 7)), false, 1.0, b)
        C4 = ncon(Any[A], Any[[-2, 1, 2, -3, 1, -1, 2]]; backend = b)
        @test C1 ≈ C2
        @test C2 == C3 == C4
        @test_throws IndexError tensortrace(
            randn(2, 2, 2, 2, 2, 2, 2), ((1,), (3, 2)), ((1, 5), (2, 6)), false
        )
    end

    @testset "tensorcontract" begin
        A = randn(Float64, (3, 20, 5, 3, 4))
        B = randn(Float64, (5, 6, 20, 3))
        C1 = zeros(3, 3, 4, 3, 6)
        for a in 1:3, b in 1:20, c in 1:5, d in 1:3, e in 1:4, f in 1:6, g in 1:3
            C1[a, g, e, d, f] += A[a, b, c, d, e] * B[c, f, b, g]
        end
        C2 = @inferred tensorcontract(
            (:a, :g, :e, :d, :f), A, (:a, :b, :c, :d, :e), B, (:c, :f, :b, :g);
            backend = b
        )
        C3 = @inferred tensorcontract(
            A, ((1, 4, 5), (2, 3)), false, B, ((3, 1), (2, 4)),
            false, ((1, 5, 3, 2, 4), ()), 1.0, b
        )
        C4 = @inferred tensorcontract(
            A, ((1, 4, 5), (2, 3)), false, B, ((3, 1), (2, 4)),
            false, ((1, 5, 3, 2, 4), ()), 1.0, b,
            ManualAllocator()
        )
        C5 = ncon(
            Any[A, B], Any[[-1, 1, 2, -4, -3], [2, -5, 1, -2]]; backend = b,
            allocator = ManualAllocator()
        )

        @test C1 ≈ C2
        @test C2 == C3 == C4 == C5
        @test_throws IndexError tensorcontract(A, [:a, :b, :c, :d], B, [:c, :f, :b, :g])
        @test_throws IndexError tensorcontract(A, [:a, :b, :c, :a, :e], B, [:c, :f, :b, :g])
    end

    @testset "tensorproduct" begin
        A = randn(Float64, (5, 5, 5, 5))
        B = rand(ComplexF64, (5, 5, 5, 5))
        C1 = kron(reshape(B, (25, 25)), reshape(A, (25, 25)))
        C2 = reshape(
            (
                @inferred tensorproduct(
                    (1, 2, 5, 6, 3, 4, 7, 8), A, (1, 2, 3, 4), B, (5, 6, 7, 8); backend = b
                )
            ),
            (5 * 5 * 5 * 5, 5 * 5 * 5 * 5)
        )
        @test C1 ≈ C2
        @test_throws IndexError tensorproduct(
            A, [:a, :b, :c, :d], B, [:d, :e, :f, :g]
        )
        @test_throws IndexError tensorproduct(
            [:a, :b, :c, :d, :e, :f, :g, :i], A, [:a, :b, :c, :d], B, [:e, :f, :g, :h]
        )

        A = rand(1, 2)
        B = rand(4, 5)
        C1 = zeros(2, 4, 1, 5)
        for i in axes(C1, 1), j in axes(C1, 2), k in axes(C1, 3), l in axes(C1, 4)
            C1[i, j, k, l] = A[k, i] * B[j, l]
        end
        C2 = tensorcontract(
            (-1, -2, -3, -4), A, (-3, -1), false, B, (-2, -4), false;
            backend = b
        )
        C3 = tensorproduct(
            A, ((1, 2), ()), false, B, ((), (1, 2)), false,
            ((2, 3, 1, 4), ()), 1.0, b
        )
        C4 = tensorproduct(
            A, ((1, 2), ()), false, B, ((), (1, 2)), false,
            ((2, 3, 1, 4), ()), 1.0, b, ManualAllocator()
        )
        @test C1 ≈ C2
        @test C2 == C3 == C4
    end
end
