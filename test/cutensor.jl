@testset "@cutensor dependency check" begin
    @test_throws ArgumentError begin
        ex = :(@cutensor A[a, b, c, d] := B[a, b, c, d])
        macroexpand(Main, ex)
    end
end

using cuTENSOR
if cuTENSOR.functional()
    using cuTENSOR: CUDACore
    using cuTENSOR.CUDACore: CuMatrix, CuArray
    using cuRAND
    using LinearAlgebra: norm
    using TensorOperations: IndexError
    using TensorOperations: cuTENSORBackend, CUDAAllocator

    @testset "elementary operations" verbose = true begin
        @testset "tensorcopy" begin
            A = randn(Float32, (3, 5, 4, 6))
            @tensor C1[4, 1, 3, 2] := A[1, 2, 3, 4]
            @tensor C2[4, 1, 3, 2] := CuArray(A)[1, 2, 3, 4]
            @test C2 isa CuArray
            @test collect(C2) ≈ C1
        end

        @testset "tensoradd" begin
            A = randn(Float32, (5, 6, 3, 4))
            B = randn(Float32, (5, 6, 3, 4))
            α = randn(Float32)
            @tensor C1[a, b, c, d] := A[a, b, c, d] + α * B[a, b, c, d]
            @tensor C2[a, b, c, d] := CuArray(A)[a, b, c, d] + α * CuArray(B)[a, b, c, d]
            @test collect(C2) ≈ C1

            C = randn(ComplexF32, (5, 6, 3, 4))
            D = randn(ComplexF32, (5, 3, 4, 6))
            β = randn(ComplexF32)
            @tensor E1[a, b, c, d] := C[a, b, c, d] + β * conj(D[a, c, d, b])
            @tensor E2[a, b, c, d] := CuArray(C)[a, b, c, d] +
                β * conj(CuArray(D)[a, c, d, b])
            @test collect(E2) ≈ E1
        end

        @testset "tensortrace" begin
            A = randn(Float32, (5, 10, 10))
            @tensor B1[a] := A[a, b′, b′]
            @tensor B2[a] := CuArray(A)[a, b′, b′]
            @test collect(B2) ≈ B1

            C = randn(ComplexF32, (3, 20, 5, 3, 20, 4, 5))
            @tensor D1[e, a, d] := C[a, b, c, d, b, e, c]
            @tensor D2[e, a, d] := CuArray(C)[a, b, c, d, b, e, c]
            @test collect(D2) ≈ D1

            @tensor D3[a, e, d] := conj(C[a, b, c, d, b, e, c])
            @tensor D4[a, e, d] := conj(CuArray(C)[a, b, c, d, b, e, c])
            @test collect(D4) ≈ D3

            α = randn(ComplexF32)
            @tensor D5[d, e, a] := α * C[a, b, c, d, b, e, c]
            @tensor D6[d, e, a] := α * CuArray(C)[a, b, c, d, b, e, c]
            @test collect(D6) ≈ D5
        end

        @testset "tensorcontract" begin
            A = randn(Float32, (3, 20, 5, 3, 4))
            B = randn(Float32, (5, 6, 20, 3))
            @tensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
            @tensor C2[a, g, e, d, f] := CuArray(A)[a, b, c, d, e] * CuArray(B)[c, f, b, g]
            @test collect(C2) ≈ C1

            D = randn(Float64, (5, 5, 5))
            E = rand(ComplexF64, (5, 5, 5))
            @tensor F1[a, b, c, d, e, f] := D[a, b, c] * conj(E[d, e, f])
            @tensor F2[a, b, c, d, e, f] := CuArray(D)[a, b, c] * conj(CuArray(E)[d, e, f])
            @test collect(F2) ≈ F1
        end
    end

    @testset "more complicated expressions" verbose = true begin
        Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
        A = rand(ComplexF64, (Dc, Da, Df, Da, De, Db, Db, Dg))
        B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
        C = rand(ComplexF64, (Dd, Dh, Df))

        @tensor D1[d, f, h] := A[c, a, f, a, e, b, b, g] * B[c, h, g, e, d] +
            0.5 * C[d, h, f]
        @tensor D2[d, f, h] := CuArray(A)[c, a, f, a, e, b, b, g] *
            CuArray(B)[c, h, g, e, d] + 0.5 * CuArray(C)[d, h, f]
        @test collect(D2) ≈ D1

        @test norm(vec(D1)) ≈ sqrt(abs(@tensor D1[d, f, h] * conj(D1[d, f, h])))
        @test norm(D2) ≈ sqrt(abs(@tensor D2[d, f, h] * conj(D2[d, f, h])))

        @testset "readme example" begin
            α = randn()
            A = randn(5, 5, 5, 5, 5, 5)
            B = randn(5, 5, 5)
            C = randn(5, 5, 5)
            D = zeros(5, 5, 5)
            D2 = CuArray(D)
            @tensor begin
                D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
                E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
            end
            @tensor begin
                D2[a, b, c] = CuArray(A)[a, e, f, c, f, g] * CuArray(B)[g, b, e] +
                    α * CuArray(C)[c, a, b]
                E2[a, b, c] := CuArray(A)[a, e, f, c, f, g] * CuArray(B)[g, b, e] +
                    α * CuArray(C)[c, a, b]
            end
            @test collect(D2) ≈ D
            @test collect(E2) ≈ E
        end

        @testset "tensor network examples ($T)" for T in
            (Float32, Float64, ComplexF32, ComplexF64)
            D1, D2, D3 = 30, 40, 20
            d1, d2 = 2, 3

            A1 = randn(T, D1, d1, D2)
            A2 = randn(T, D2, d2, D3)
            ρₗ = randn(T, D1, D1)
            ρᵣ = randn(T, D3, D3)
            H = randn(T, d1, d2, d1, d2)

            @tensor begin
                HRAA1[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @tensor begin
                HRAA2[a, s1, s2, c] := CuArray(ρₗ)[a, a'] * CuArray(A1)[a', t1, b] *
                    CuArray(A2)[b, t2, c'] * CuArray(ρᵣ)[c', c] * CuArray(H)[s1, s2, t1, t2]
            end
            @test HRAA2 isa CuArray{T}
            @test collect(HRAA2) ≈ HRAA1

            cumemtypes = (CUDACore.DeviceMemory, CUDACore.UnifiedMemory, CUDACore.HostMemory)
            for Mout in cumemtypes
                Min = CUDACore.DeviceMemory
                Mtemp = CUDACore.DeviceMemory
                allocator = CUDAAllocator{Mout, Min, Mtemp}()
                @tensor backend = cuTENSORBackend() allocator = allocator begin
                    HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                        ρᵣ[c', c] * H[s1, s2, t1, t2]
                end
                @test HRAA3 isa CuArray{T, 4, Mout}
                @test collect(HRAA3) ≈ HRAA1
            end
            for Min in cumemtypes
                Mout = CUDACore.UnifiedMemory
                Mtemp = CUDACore.UnifiedMemory
                allocator = CUDAAllocator{Mout, Min, Mtemp}()
                @tensor backend = cuTENSORBackend() allocator = allocator begin
                    HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                        ρᵣ[c', c] * H[s1, s2, t1, t2]
                end
                @test HRAA3 isa CuArray{T, 4, Mout}
                @test collect(HRAA3) ≈ HRAA1
            end

            @tensor begin
                E1 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] * H[t, t', s, s'] *
                    conj(A1[a', t, b']) * conj(A2[b', t', c'])
                E2 = CuArray(ρₗ)[a', a] * CuArray(A1)[a, s, b] * CuArray(A2)[b, s', c] *
                    CuArray(ρᵣ)[c, c'] * CuArray(H)[t, t', s, s'] *
                    conj(CuArray(A1)[a', t, b']) * conj(CuArray(A2)[b', t', c'])
            end
            @tensor backend = cuTENSORBackend() allocator = CUDAAllocator() begin
                E3 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] * H[t, t', s, s'] *
                    conj(A1[a', t, b']) * conj(A2[b', t', c'])
            end
            @test E1 ≈ E2 ≈ E3
        end
    end

    @testset "@cutensor" verbose = true begin
        @testset "tensorcontract 1" begin
            A = randn(Float64, (3, 5, 4, 6))
            @tensor C1[4, 1, 3, 2] := A[1, 2, 3, 4]
            @cutensor C2[4, 1, 3, 2] := A[1, 2, 3, 4]
            @test C1 ≈ collect(C2)
            @test_throws IndexError begin
                @cutensor C[1, 2, 3, 4] := A[1, 2, 3]
            end
            @test_throws IndexError begin
                @cutensor C[1, 2, 3, 4] := A[1, 2, 2, 4]
            end

            B = randn(Float64, (5, 6, 3, 4))
            p = [3, 1, 4, 2]
            @tensor C1[3, 1, 4, 2] := A[3, 1, 4, 2] + B[1, 2, 3, 4]
            @cutensor C2[3, 1, 4, 2] := A[3, 1, 4, 2] + B[1, 2, 3, 4]
            @test C1 ≈ collect(C2)
            @test_throws CUTENSORError begin
                @cutensor C[1, 2, 3, 4] := A[1, 2, 3, 4] + B[1, 2, 3, 4]
            end

            A = randn(Float64, (50, 100, 100))
            @tensor C1[a] := A[a, b', b']
            @cutensor C2[a] := A[a, b', b']
            @test C1 ≈ collect(C2)

            A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
            @tensor C1[e, a, d] := A[a, b, c, d, b, e, c]
            @cutensor C2[e, a, d] := A[a, b, c, d, b, e, c]
            @test C1 ≈ collect(C2)

            A = randn(Float64, (3, 20, 5, 3, 4))
            B = randn(Float64, (5, 6, 20, 3))
            @tensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
            @cutensor C2[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
            @test C1 ≈ collect(C2)
            @test_throws IndexError begin
                @cutensor A[a, b, c, d] * B[c, f, b, g]
            end
        end

        @testset "tensorcontract 2" begin
            A = randn(Float64, (5, 5, 5, 5))
            B = rand(ComplexF64, (5, 5, 5, 5))
            @tensor C1[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
            @cutensor C2[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
            @test C1 ≈ collect(C2)
            @test_throws IndexError begin
                @cutensor C[a, b, c, d, e, f, g, i] := A[a, b, c, d] * B[e, f, g, h]
            end
        end

        @testset "tensorcontract 3" begin
            Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
            A = rand(ComplexF64, (Da, Dc, Df, Da, De, Db, Db, Dg))
            B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
            C = rand(ComplexF64, (Dd, Dh, Df))
            @tensor D1[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] +
                0.5 * C[d, h, f]
            @cutensor D2[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] +
                0.5 * C[d, h, f]
            @test D1 ≈ collect(D2)
            E1 = sqrt(abs((@tensor tensorscalar(D1[d, f, h] * conj(D1[d, f, h])))))
            E2 = sqrt(abs((@cutensor tensorscalar(D2[d, f, h] * conj(D2[d, f, h])))))
            @test E1 ≈ E2
        end

        @testset "views" begin
            p = [3, 1, 4, 2]
            Abig = cuRAND.randn(Float32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8)
            )
            Cbig = CUDACore.zeros(Float32, (50, 50, 50, 50))
            C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
            Acopy = copy(A)
            Ccopy = copy(C)
            @tensor C[3, 1, 4, 2] = A[1, 2, 3, 4]
            @tensor Ccopy[3, 1, 4, 2] = Acopy[1, 2, 3, 4]
            @test copy(C) ≈ Ccopy
            @test_throws TensorOperations.IndexError begin
                @tensor C[3, 1, 4, 2] = A[1, 2, 3]
            end
            @test_throws CUTENSORError begin
                @tensor C[3, 1, 4, 2] = A[3, 1, 4, 2]
            end
            @test_throws TensorOperations.IndexError begin
                @tensor C[1, 1, 2, 3] = A[1, 2, 3, 4]
            end
        end

        @testset "views 2" begin
            p = [3, 1, 4, 2]
            Abig = cuRAND.randn(ComplexF32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8)
            )
            Cbig = CUDACore.zeros(ComplexF32, (50, 50, 50, 50))
            C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
            Acopy = permutedims(copy(A), p)
            Ccopy = copy(C)
            α = randn(Float64)
            β = randn(Float64)
            @tensor C[3, 1, 4, 2] = β * C[3, 1, 4, 2] + α * A[1, 2, 3, 4]
            Ccopy = β * Ccopy + α * Acopy
            @test copy(C) ≈ Ccopy
            @test_throws IndexError begin
                @tensor C[3, 1, 4, 2] = 0.5 * C[3, 1, 4, 2] + 1.2 * A[1, 2, 3]
            end
            @test_throws CUTENSORError begin
                @tensor C[3, 1, 4, 2] = 0.5 * C[3, 1, 4, 2] + 1.2 * A[3, 1, 2, 4]
            end
            @test_throws IndexError begin
                @tensor C[1, 1, 2, 3] = 0.5 * C[1, 1, 2, 3] + 1.2 * A[1, 2, 3, 4]
            end
        end

        @testset "views 3" begin
            Abig = cuRAND.rand(ComplexF64, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8)
            )
            Bbig = cuRAND.rand(ComplexF64, (50, 50))
            B = view(Bbig, 13 .+ (0:14), 3 .+ 5 .* (0:6))
            Acopy = copy(A)
            Bcopy = copy(B)
            α = randn(Float64)
            @tensor B[b, c] += α * A[a, b, c, a]
            @tensor Bcopy[b, c] += α * Acopy[a, b, c, a]
            @test copy(B) ≈ Bcopy
            @test_throws IndexError begin
                @tensor B[b, c] += α * A[a, b, c]
            end
            @test_throws CUTENSORError begin
                @tensor B[c, b] += α * A[a, b, c, a]
            end
            @test_throws IndexError begin
                @tensor B[c, b] += α * A[a, b, a, a]
            end
            @test_throws CUTENSORError begin
                @tensor B[c, b] += α * A[a, b, a, c]
            end
        end

        @testset "views 4" begin
            Abig = CUDACore.rand(ComplexF32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6),
                7 .+ 2 .* (0:8)
            )
            Bbig = CUDACore.rand(ComplexF32, (50, 50, 50))
            B = view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
            Cbig = CUDACore.rand(ComplexF32, (40, 40, 40))
            C = view(Cbig, 3 .+ 2 .* (0:8), 13 .+ (0:8), 7 .+ 3 .* (0:7))
            Acopy = copy(A)
            Bcopy = copy(B)
            Ccopy = copy(C)
            α = randn(Float64)
            @tensor C[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
            @tensor Ccopy[d, a, e] -= α * Acopy[a, b, c, d] * conj(Bcopy[c, e, b])
            @test copy(C) ≈ Ccopy
        end
    end

    @testset "Issues" verbose = true begin
        @testset "Issue PR #186" begin
            # https://github.com/Jutho/TensorOperations.jl/pull/186
            A = randn(Float32, (5, 5, 5, 5))
            Atr = @tensor A[a, b, b, a]
            Atr2 = @cutensor A[a, b, b, a]
            @test Atr ≈ Atr2
        end
    end
end
