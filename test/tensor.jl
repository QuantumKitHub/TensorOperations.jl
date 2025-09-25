# test index notation using @tensor macro
#-----------------------------------------
backendlist = (BaseCopy(), BaseView(), StridedNative(), StridedBLAS())

@testset "with backend $b" for b in backendlist
    @testset "tensorcontract 1" begin
        A = randn(Float64, (3, 5, 4, 6))
        p = (4, 1, 3, 2)
        C1 = permutedims(A, p)
        @tensor backend = b C2[4, 1, 3, 2] := A[1, 2, 3, 4]
        @test C1 ≈ C2
        @test_throws IndexError begin
            @tensor C[1, 2, 3, 4] := A[1, 2, 3]
        end
        @test_throws IndexError begin
            @tensor C[1, 2, 3, 4] := A[1, 2, 2, 4]
        end

        B = randn(Float64, (5, 6, 3, 4))
        p = [3, 1, 4, 2]
        @tensor backend = b C1[3, 1, 4, 2] := A[3, 1, 4, 2] + B[1, 2, 3, 4]
        C2 = A + permutedims(B, p)
        @test C1 ≈ C2
        @test_throws DimensionMismatch begin
            @tensor backend = b C[1, 2, 3, 4] := A[1, 2, 3, 4] + B[1, 2, 3, 4]
        end

        A = randn(Float64, (50, 100, 100))
        @tensor backend = b C1[a] := A[a, b', b']
        C2 = zeros(50)
        for i in 1:50
            for j in 1:100
                C2[i] += A[i, j, j]
            end
        end
        @test C1 ≈ C2
        A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
        @tensor backend = b C1[e, a, d] := A[a, b, c, d, b, e, c]
        C2 = zeros(4, 3, 3)
        for i1 in 1:4, i2 in 1:3, i3 in 1:3
            for j1 in 1:20, j2 in 1:5
                C2[i1, i2, i3] += A[i2, j1, j2, i3, j1, i1, j2]
            end
        end
        @test C1 ≈ C2

        A = randn(Float64, (3, 20, 5, 3, 4))
        B = randn(Float64, (5, 6, 20, 3))
        @tensor backend = b C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
        C2 = zeros(3, 3, 4, 3, 6)
        for a in 1:3, b in 1:20, c in 1:5, d in 1:3, e in 1:4, f in 1:6, g in 1:3
            C2[a, g, e, d, f] += A[a, b, c, d, e] * B[c, f, b, g]
        end
        @test C1 ≈ C2
        @test_throws IndexError begin
            @tensor backend = b A[a, b, c, d] * B[c, f, b, g]
        end
    end

    @testset "tensorcontract 2" begin
        A = randn(Float64, (5, 5, 5, 5))
        B = rand(ComplexF64, (5, 5, 5, 5))
        @tensor backend = b C1[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
        C2 = reshape(
            kron(reshape(B, (25, 25)), reshape(A, (25, 25))),
            (5, 5, 5, 5, 5, 5, 5, 5)
        )
        @test C1 ≈ C2
        @test_throws IndexError begin
            @tensor backend = b C[a, b, c, d, e, f, g, i] := A[a, b, c, d] * B[e, f, g, h]
        end
    end

    @testset "tensorcontract 3" begin
        Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
        A = rand(ComplexF64, (Da, Dc, Df, Da, De, Db, Db, Dg))
        B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
        C = rand(ComplexF64, (Dd, Dh, Df))
        @tensor backend = b D1[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] +
            0.5 * C[d, h, f]
        D2 = zeros(ComplexF64, (Dd, Df, Dh))
        for d in 1:Dd, f in 1:Df, h in 1:Dh
            D2[d, f, h] += 0.5 * C[d, h, f]
            for a in 1:Da, b in 1:Db, c in 1:Dc, e in 1:De, g in 1:Dg
                D2[d, f, h] += A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d]
            end
        end
        @test D1 ≈ D2
        @test norm(vec(D1)) ≈
            sqrt(abs((@tensor backend = b tensorscalar(D1[d, f, h] * conj(D1[d, f, h])))))
    end

    @testset "views" begin
        p = [3, 1, 4, 2]
        Abig = randn(Float64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8))
        Cbig = zeros(ComplexF64, (50, 50, 50, 50))
        C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
        Acopy = copy(A)
        Ccopy = copy(C)
        @tensor backend = b C[3, 1, 4, 2] = A[1, 2, 3, 4]
        @tensor backend = b Ccopy[3, 1, 4, 2] = Acopy[1, 2, 3, 4]
        @test C ≈ Ccopy
        @test_throws IndexError begin
            @tensor C[3, 1, 4, 2] = A[1, 2, 3]
        end
        @test_throws DimensionMismatch begin
            @tensor C[3, 1, 4, 2] = A[3, 1, 4, 2]
        end
        @test_throws IndexError begin
            @tensor C[1, 1, 2, 3] = A[1, 2, 3, 4]
        end
    end

    @testset "views 2" begin
        p = [3, 1, 4, 2]
        Abig = randn(Float64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8))
        Cbig = zeros(ComplexF64, (50, 50, 50, 50))
        C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
        Acopy = permutedims(copy(A), p)
        Ccopy = copy(C)
        α = randn(Float64)
        β = randn(Float64)
        @tensor backend = b C[3, 1, 4, 2] = β * C[3, 1, 4, 2] + α * A[1, 2, 3, 4]
        Ccopy = β * Ccopy + α * Acopy
        @test C ≈ Ccopy
        @test_throws IndexError begin
            @tensor C[3, 1, 4, 2] = 0.5 * C[3, 1, 4, 2] + 1.2 * A[1, 2, 3]
        end
        @test_throws DimensionMismatch begin
            @tensor C[3, 1, 4, 2] = 0.5 * C[3, 1, 4, 2] + 1.2 * A[3, 1, 2, 4]
        end
        @test_throws IndexError begin
            @tensor C[1, 1, 2, 3] = 0.5 * C[1, 1, 2, 3] + 1.2 * A[1, 2, 3, 4]
        end
    end

    @testset "views 3" begin
        Abig = rand(Float64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
        Bbig = rand(ComplexF64, (50, 50))
        B = view(Bbig, 13 .+ (0:14), 3 .+ 5 .* (0:6))
        Acopy = copy(A)
        Bcopy = copy(B)
        α = randn(Float64)
        @tensor backend = b B[b, c] += α * A[a, b, c, a]
        for i in 1 .+ (0:8)
            Bcopy += α * view(A, i, :, :, i)
        end
        @test B ≈ Bcopy
        @test_throws IndexError begin
            @tensor B[b, c] += α * A[a, b, c]
        end
        @test_throws DimensionMismatch begin
            @tensor B[c, b] += α * A[a, b, c, a]
        end
        @test_throws IndexError begin
            @tensor B[c, b] += α * A[a, b, a, a]
        end
        @test_throws DimensionMismatch begin
            @tensor B[c, b] += α * A[a, b, a, c]
        end
    end

    @testset "views 4" begin
        Abig = rand(Float64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
        Bbig = rand(ComplexF64, (50, 50, 50))
        B = view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
        Cbig = rand(ComplexF32, (40, 40, 40))
        C = view(Cbig, 3 .+ 2 .* (0:8), 13 .+ (0:8), 7 .+ 3 .* (0:7))
        Acopy = copy(A)
        Bcopy = copy(B)
        Ccopy = copy(C)
        α = randn(Float64)
        for d in 1 .+ (0:8), a in 1 .+ (0:8), e in 1 .+ (0:7)
            for b in 1 .+ (0:14), c in 1 .+ (0:6)
                Ccopy[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
            end
        end
        @tensor backend = b C[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
        @test C ≈ Ccopy
    end

    @testset "Float32 views" begin
        α = randn(Float64)
        Abig = rand(Float64, (30, 30, 30, 30))
        A = view(Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8))
        Bbig = rand(ComplexF64, (50, 50, 50))
        B = view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
        Cbig = rand(ComplexF32, (40, 40, 40))
        C = view(Cbig, 3 .+ 2 .* (0:8), 13 .+ (0:8), 7 .+ 3 .* (0:7))
        Ccopy = copy(C)
        for d in 1 .+ (0:8), a in 1 .+ (0:8), e in 1 .+ (0:7)
            for b in 1 .+ (0:14), c in 1 .+ (0:6)
                Ccopy[d, a, e] += α * A[a, b, c, d] * conj(B[c, e, b])
            end
        end
        @tensor backend = b C[d, a, e] += α * A[a, b, c, d] * conj(B[c, e, b])
        @test C ≈ Ccopy
        @test_throws IndexError begin
            @tensor C[d, a, e] += α * A[a, b, c, a] * B[c, e, b]
        end
        @test_throws IndexError begin
            @tensor C[d, a, e] += α * A[a, b, c, d] * B[c, b]
        end
        @test_throws IndexError begin
            @tensor C[d, e] += α * A[a, b, c, d] * B[c, e, b]
        end
        @test_throws DimensionMismatch begin
            @tensor C[d, e, a] += α * A[a, b, c, d] * B[c, e, b]
        end
    end

    # Simple function example
    # @tensor function f(A, b)
    #     w[x] := (1 // 2) * A[x, y] * b[y]
    #     return w
    # end
    # for T in (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
    #     A = rand(T, 10, 10)
    #     b = rand(T, 10)
    #     @test f(A, b) ≈ (1 // 2) * A * b
    # end

    # Example from README.md
    @testset "README example" begin
        α = randn()
        A = randn(5, 5, 5, 5, 5, 5)
        B = randn(5, 5, 5)
        C = randn(5, 5, 5)
        D = zeros(5, 5, 5)
        @tensor begin
            D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
            E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
        end
        @test D == E
    end

    # Some tensor network examples
    if !(b isa StridedBLAS)
        scalartypelist = (Float32, Float64, ComplexF32, ComplexF64, BigFloat)
    else
        scalartypelist = (Float32, Float64, ComplexF32, ComplexF64)
    end
    @testset "tensor network examples ($T)" for T in scalartypelist
        D1, D2, D3 = 30, 40, 20
        d1, d2 = 2, 3
        A1 = rand(T, D1, d1, D2) .- 1 // 2
        A2 = rand(T, D2, d2, D3) .- 1 // 2
        rhoL = rand(T, D1, D1) .- 1 // 2
        rhoR = rand(T, D3, D3) .- 1 // 2
        H = rand(T, d1, d2, d1, d2) .- 1 // 2
        A12 = reshape(reshape(A1, D1 * d1, D2) * reshape(A2, D2, d2 * D3), (D1, d1, d2, D3))
        rA12 = reshape(
            reshape(rhoL * reshape(A12, (D1, d1 * d2 * D3)), (D1 * d1 * d2, D3)) * rhoR,
            (D1, d1, d2, D3)
        )
        HrA12 = permutedims(
            reshape(
                reshape(H, (d1 * d2, d1 * d2)) *
                    reshape(permutedims(rA12, (2, 3, 1, 4)), (d1 * d2, D1 * D3)),
                (d1, d2, D1, D3)
            ),
            (3, 1, 2, 4)
        )
        E = dot(A12, HrA12)
        @tensor backend = b HrA12′[a, s1, s2, c] := rhoL[a, a'] * A1[a', t1, b] *
            A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]
        @tensor backend = b HrA12′′[:] := rhoL[-1, 1] * H[-2, -3, 4, 5] * A2[2, 5, 3] *
            rhoR[3, -4] * A1[1, 4, 2] # should be contracted in exactly same order
        @tensor backend = b order = (a', b, c', t1, t2) begin
            HrA12′′′[a, s1, s2, c] := rhoL[a, a'] * H[s1, s2, t1, t2] * A2[b, t2, c'] *
                rhoR[c', c] * A1[a', t1, b] # should be contracted in exactly same order
        end
        @tensoropt HrA12′′′′[:] := rhoL[-1, 1] * H[-2, -3, 4, 5] * A2[2, 5, 3] *
            rhoR[3, -4] * A1[1, 4, 2]

        @test HrA12′ == HrA12′′ == HrA12′′′ # should be exactly equal
        @test HrA12 ≈ HrA12′
        @test HrA12 ≈ HrA12′′′′
        @test HrA12′′ ≈ ncon(
            [rhoL, H, A2, rhoR, A1],
            [[-1, 1], [-2, -3, 4, 5], [2, 5, 3], [3, -4], [1, 4, 2]]
        )
        @test HrA12′′ == @ncon(
            [rhoL, H, A2, rhoR, A1],
            [[-1, 1], [-2, -3, 4, 5], [2, 5, 3], [3, -4], [1, 4, 2]];
            order = [1, 2, 3, 4, 5], output = [-1, -2, -3, -4], backend = b
        )
        @test E ≈
            @tensor tensorscalar(
            rhoL[a', a] * A1[a, s, b] * A2[b, s', c] * rhoR[c, c'] *
                H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c'])
        )
        @test E ≈ @ncon(
            [rhoL, A1, A2, rhoR, H, conj(A1), conj(A2)],
            [[5, 1], [1, 2, 3], [3, 4, 9], [9, 10], [6, 8, 2, 4], [5, 6, 7], [7, 8, 10]]
        )
        # this implicitly tests that `ncon` returns a scalar if no open indices
    end
end
@testset "tensoropt" begin
    A = randn(5, 5, 5, 5)
    B = randn(5, 5, 5)
    C = randn(5, 5, 5)
    @tensoropt (a => χ, b => χ^2, c => 2 * χ, d => χ, e => 5, f => 2 * χ) begin
        D1[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensoropt (a = χ, b = χ^2, c = 2 * χ, d = χ, e = 5, f = 2 * χ) begin
        D2[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensoropt ((a, d) => χ, b => χ^2, (c, f) => 2 * χ, e => 5) begin
        D3[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @tensoropt ((a, d) = χ, b = χ^2, (c, f) = 2 * χ, e = 5) begin
        D4[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b]
    end
    @test D1 == D2 == D3 == D4
    _optdata = optex -> TensorOperations.optdata(
        optex,
        :(D1[a, b, c, d] := A[a, e, c, f] * B[g, d, e] * C[g, f, b])
    )
    optex1 = :((a => χ, b => χ^2, c => 2 * χ, d => χ, e => 5, f => 2 * χ))
    optex2 = :((a = χ, b = χ^2, c = 2 * χ, d = χ, e = 5, f = 2 * χ))
    optex3 = :(((a, d) => χ, b => χ^2, (c, f) => 2 * χ, e => 5))
    optex4 = :(((a, d) = χ, b = χ^2, (c, f) = 2 * χ, e = 5))
    optex5 = :(
        ((a,) => χ, b => χ^2, (c,) => 2χ, d => χ, e => 5, f => χ * 2, () => 12345)
    )
    @test _optdata(optex1) == _optdata(optex2) == _optdata(optex3) ==
        _optdata(optex4) == _optdata(optex5)
    optex6 = :(((a, b, c) = χ,))
    optex7 = :((a, b, c))
    @test _optdata(optex6) == _optdata(optex7)
    optex8 = :(((a, b, c) = 1, (d, e, f, g) = χ))
    optex9 = :(!(a, b, c))
    @test _optdata(optex8) == _optdata(optex9)
    optex10 = :((a => χ, b => χ^2, c = 2 * χ, d => χ, e => 5, f = 2 * χ))
    optex11 = :((a = χ, b = χ^2, c = 2 * χ, d, e = 5, f))
    @test_throws ErrorException _optdata(optex10)
    @test_throws ErrorException _optdata(optex11)
end

# diagonal matrices
@testset "Diagonal($T)" for T in (Float32, Float64, ComplexF32, ComplexF64)
    A = randn(T, 10, 10, 10, 10)
    @tensor C[3, 1, 4, 2] := A[1, 2, 3, 4]
    U, S, V = svd(reshape(C, (100, 100)))
    U2 = reshape(view(U, 1:2:100, :), (5, 10, 100))
    S2 = Diagonal(S)
    V2 = reshape(view(V, 1:2:100, :), (5, 10, 100))
    @tensor A2[1, 2, 3, 4] := U2[3, 1, a] * S2[a, a'] * conj(V2[4, 2, a'])
    @test A2 ≈ view(A, :, :, 1:2:10, 1:2:10)

    S = Diagonal(randn(T, 5))
    @tensor S2[i, k] := S[i, j] * S[k, j]
    S3 = similar(S)
    @tensor S3[i, k] = S[i, j] * S[j, k]
    @test S2 ≈ S3 ≈ S * S
    Str = @tensor S[i, j] * S[i, j]
    @test Str ≈ sum(S.diag .^ 2)

    @tensor SS[i, j, k, l] := S[i, j] * S[k, l]
    @tensor SS2[i, j, k, l] := Array(S)[i, j] * Array(S)[k, l]
    @test SS ≈ SS2

    B = randn(T, 10)
    @tensor C1[3, 1, 4, 2] := A[a, 2, 3, 4] * Diagonal(B)[a, 1]
    @tensor C2[3, 1, 4, 2] := A[1, a, 3, 4] * conj(Diagonal(B)[a, 2])
    @tensor C3[3, 1, 4, 2] := conj(A[1, 2, a, 4]) * Diagonal(B)[a, 3]
    @tensor C4[3, 1, 4, 2] := conj(A[1, 2, 3, a]) * conj(Diagonal(B)[a, 4])
    @tensor C1′[3, 1, 4, 2] := Diagonal(B)[a, 1] * A[a, 2, 3, 4]
    @tensor C2′[3, 1, 4, 2] := conj(Diagonal(B)[a, 2]) * A[1, a, 3, 4]
    @tensor C3′[3, 1, 4, 2] := Diagonal(B)[a, 3] * conj(A[1, 2, a, 4])
    @tensor C4′[3, 1, 4, 2] := conj(Diagonal(B)[a, 4]) * conj(A[1, 2, 3, a])
    @test C1 == C1′
    @test C2 == C2′
    @test C3 == C3′
    @test C4 == C4′
    for i in 1:10
        @test C1[:, i, :, :] ≈ permutedims(A[i, :, :, :], (2, 3, 1)) * B[i]
        @test C2[:, :, :, i] ≈ permutedims(A[:, i, :, :], (2, 1, 3)) * conj(B[i])
        @test C3[i, :, :, :] ≈ conj(permutedims(A[:, :, i, :], (1, 3, 2))) * B[i]
        @test C4[:, :, i, :] ≈ conj(permutedims(A[:, :, :, i], (3, 1, 2))) * conj(B[i])
    end

    @tensor D1[1, 2] := A[1, 2, a, b] * Diagonal(B)[a, b]
    @tensor D2[1, 2] := A[1, b, 2, a] * conj(Diagonal(B)[a, b])
    @tensor D3[1, 2] := conj(A[a, 2, 1, b]) * Diagonal(B)[a, b]
    @tensor D4[1, 2] := conj(A[a, 1, b, 2]) * conj(Diagonal(B)[a, b])
    @tensor D1′[1, 2] := Diagonal(B)[a, b] * A[1, 2, a, b]
    @tensor D2′[1, 2] := conj(Diagonal(B)[a, b]) * A[1, b, 2, a]
    @tensor D3′[1, 2] := Diagonal(B)[a, b] * conj(A[a, 2, 1, b])
    @tensor D4′[1, 2] := conj(Diagonal(B)[a, b]) * conj(A[a, 1, b, 2])
    @test D1 == D1′
    @test D2 == D2′
    @test D3 == D3′
    @test D4 == D4′

    E1 = zero(D1)
    E2 = zero(D2)
    E3 = zero(D3)
    E4 = zero(D4)
    for i in 1:10
        E1[:, :] += A[:, :, i, i] * B[i]
        E2[:, :] += A[:, i, :, i] * conj(B[i])
        E3[:, :] += A[i, :, :, i]' * B[i]
        E4[:, :] += conj(A[i, :, i, :]) * conj(B[i])
    end
    @test D1 ≈ E1
    @test D2 ≈ E2
    @test D3 ≈ E3
    @test D4 ≈ E4

    F = randn(T, (10, 10))
    @tensor G[a, c, b, d] := F[a, b] * Diagonal(B)[c, d]
    @test reshape(G, (100, 100)) ≈ kron(Diagonal(B), F)
end

@testset "Specific issues" begin
    @testset "Issue 83" begin
        op1 = randn(2, 2)
        op2 = randn(2, 2)
        op3 = randn(2, 2)

        f83(op, op3) = @ncon((op, op3), ([-1 3], [3 -3]))

        b = f83(op1, op3)
        bcopy = deepcopy(b)
        c = f83(op2, op3)
        @test b == bcopy
        @test b != c
    end

    @testset "Issue 133" begin
        vec = [1, 2]
        mat1 = rand(2, 2)
        mat2 = rand(2, 2)
        @tensor res[mu] := -(mat1[mu, alpha] * vec[alpha] + mat2[mu, alpha] * vec[alpha])
        @test res ≈ -(mat1 * vec + mat2 * vec)
    end

    @testset "Issue 148" for T in (Float64, ComplexF64)
        A = randn(T, (5, 6))
        B = randn(T, (3, 4))
        X = randn(T, (6, 4))
        E = zeros(T, (5, 3))

        @tensor begin
            E[i_1, i_2] = X[j_1, j_2] * A[i_1, j_1] * B[i_2, j_2]
        end

        @test E ≈ A * X * transpose(B)
    end

    @testset "Issue 149" for T in (Float64, ComplexF64)
        v1 = rand(2)
        v2 = rand(2)
        m1 = rand(2, 2, 2)
        m2 = rand(2, 2, 2)

        res1 = @tensor begin
            (v1[a] * v1[b] * v1[c]) * m1[a, b, c] + (v2[a] * v2[b] * v2[c]) * m2[a, b, c]
        end
        res2 = @tensor (v1[a] * v1[b] * v1[c]) * m1[a, b, c]
        res3 = @tensor (v2[a] * v2[b] * v2[c]) * m2[a, b, c]
        @test res1 ≈ res2 + res3
    end

    @testset "Handling definitions" begin
        vec = rand(2)
        veccopy = copy(vec)
        mat1 = rand(2, 2)
        mat2 = rand(2, 2)
        @tensor vec[a] := (mat1[a, b] * mat2[b, c]) * vec[c]
        @test vec ≈ mat1 * mat2 * veccopy
    end

    @testset "tensorscalar expressions" begin
        mat1 = rand(ComplexF64, 2, 2)
        mat2 = rand(2, 2)
        mat3 = rand(2, 2)
        mat4 = rand(Float32, 2, 2)
        @tensoropt res1[a, b] := mat1[x, y] * mat2[y, x] * mat3[a, c] * mat4[c, b]
        @tensor res2[a, b] := (mat1[x, y] * mat2[y, x]) * mat3[a, c] * mat4[c, b]
        @tensor res3[a, b] := tensorscalar(mat1[x, y] * mat2[y, x]) * mat3[a, c] *
            mat4[c, b]
        @tensor costcheck = warn res4[a, b] := tensorscalar(mat1[x, y] * mat2[y, x]) *
            mat3[a, c] * mat4[c, b]
        @test res1 == res2 == res3 == res4

        @tensor res1[a, b] := mat1[x, y] * mat2[y, x] * mat3[b, a] + mat4[a, b]
        @tensor res2[a, b] := (mat1[x, y] * mat2[y, x]) * mat3[b, a] + mat4[a, b]
        @tensoropt res3[a, b] := mat4[a, b] + mat1[x, y] * mat3[b, a] * mat2[y, x]
        @tensor costcheck = warn res4[a, b] := mat4[a, b] +
            tensorscalar(mat1[x, y] * mat2[y, x]) * mat3[b, a]
        @test res1 == res2 == res3 == res4

        @tensor res1[a, b] := 0.5 * mat1[x, y] * mat2[y, x] * mat3[b, a]
        @tensor res2[a, b] := (mat1[x, y] * mat2[y, x]) * mat3[b, a] * 0.5
        @tensoropt res3[a, b] := 0.5 * mat1[x, y] * mat3[b, a] * mat2[y, x]
        @tensor costcheck = warn res4[a, b] := 0.5 * tensorscalar(mat1[x, y] * mat2[y, x]) *
            mat3[b, a]
        @test res1 == res2 == res3 == res4
    end

    # @testset "Issue 136" begin
    #     A = rand(2, 2)
    #     B = rand(2, 2)
    #     @test_throws ArgumentError begin
    #         @tensor A[a, b] = A[b, a]
    #     end
    #     @test_throws ArgumentError begin
    #         @tensor A[a, b] = A[a, b] + B[a, c] * A[c, b]
    #     end
    #     @test_throws ArgumentError begin
    #         @tensor A[a, b] = B[a, c] * A[c, b] + A[a, b]
    #     end
    # end

    @testset "ncon with conj" begin
        A = rand(ComplexF64, 2, 2, 2, 2)
        B = zeros(ComplexF64, 2, 2)
        for i in axes(A, 2), j in axes(A, 4)
            for k in axes(A, 1)
                B[i, j] += A[k, i, k, j]
            end
        end
        B2 = ncon([A], [[1, -1, 1, -2]], [false])
        @test B2 ≈ B
        B3 = ncon([A], [[1, -1, 1, -2]], [true])
        @test B3 ≈ conj(B)

        C = permutedims(A, (1, 4, 2, 3))
        C2 = ncon([A], [[-1, -3, -4, -2]], [false])
        @test C2 ≈ C
        C3 = ncon([A], [[-1, -3, -4, -2]], [true])
        @test C3 ≈ conj(C)
    end

    @testset "methods for StridedBLAS" begin
        using TensorOperations: isblascontractable, isblasdestination, makeblascontractable,
            StridedBLAS, DefaultAllocator
        using Strided
        backend = StridedBLAS()
        allocator = DefaultAllocator()

        A = StridedView(rand(ComplexF64, 4, 4, 4, 4))

        p = ((1, 2), (3, 4))
        @test isblascontractable(A, p)
        Anew, pnew, flag = makeblascontractable(A, p, ComplexF64, backend, allocator)
        @test isblascontractable(Anew, pnew)
        @test Anew === A

        @test !isblascontractable(conj(A), p)
        Anew, pnew, flag = makeblascontractable(A, p, ComplexF64, backend, allocator)
        @test isblascontractable(Anew, pnew)

        for p in (((2, 1), (3, 4)), ((1,), (3, 2, 4)), ((2, 1, 4), (3,))),
                op in (identity, conj)

            @test !isblascontractable(op(A), p)
            Anew, pnew, flag = makeblascontractable(
                op(A), p, ComplexF64, backend, allocator
            )
            @test isblascontractable(Anew, pnew)
        end

        vA = view(A, 1:2, 1:2, 1:2, 1:2)
        p = ((1, 2), (3, 4))
        @test !isblascontractable(vA, p)
        Anew, pnew, flag = makeblascontractable(vA, p, ComplexF64, backend, allocator)
        @test !flag
        @test isblascontractable(Anew, pnew)

        pA = permutedims(A, (3, 4, 1, 2))
        p = ((1, 2), (3, 4))
        @test isblascontractable(pA, p)
        @test isblascontractable(conj(pA), p)
    end

    @testset "Issue 220" begin
        A = rand(2, 2)
        B = rand(2, 2)
        C = rand(2, 2)
        D = rand(2, 2)
        c = 1im
        @tensor E[a; c] := c * (A[a b] * B[b c] + C[a b] * D[b c])
        @test scalartype(E) == ComplexF64
        @test E ≈ c * (A * B + C * D)
    end
end
