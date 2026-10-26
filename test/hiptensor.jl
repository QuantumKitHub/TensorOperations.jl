@testset "@hiptensor dependency check" begin
    @test_throws ArgumentError begin
        ex = :(@hiptensor A[a, b, c, d] := B[a, b, c, d])
        macroexpand(Main, ex)
    end
end

using AMDGPU, AMDGPU.hipTENSOR
using AMDGPU.hipTENSOR: hipTENSOR
if hipTENSOR.has_hiptensor()
    using LinearAlgebra: norm
    using TensorOperations: IndexError
    using TensorOperations: hipTENSORBackend, AMDAllocator
    using TensorOperations: BufferAllocator, AMDBufferAllocator
    using TensorOperations: tensoralloc, tensorfree!
    using TensorOperations: allocator_checkpoint!, allocator_reset!

    @testset "elementary operations" verbose = true begin
        @testset "tensorcopy" begin
            A = randn(Float32, (3, 5, 4, 6))
            @tensor C1[4, 1, 3, 2] := A[1, 2, 3, 4]
            @tensor C2[4, 1, 3, 2] := ROCArray(A)[1, 2, 3, 4]
            @test C2 isa ROCArray
            @test collect(C2) ≈ C1
        end

        @testset "tensoradd" begin
            A = randn(Float32, (5, 6, 3, 4))
            B = randn(Float32, (5, 6, 3, 4))
            α = randn(Float32)
            @tensor C1[a, b, c, d] := A[a, b, c, d] + α * B[a, b, c, d]
            @tensor C2[a, b, c, d] := ROCArray(A)[a, b, c, d] + α * ROCArray(B)[a, b, c, d]
            @test collect(C2) ≈ C1

            C = randn(ComplexF32, (5, 6, 3, 4))
            D = randn(ComplexF32, (5, 3, 4, 6))
            β = randn(ComplexF32)
            @tensor E1[a, b, c, d] := C[a, b, c, d] + β * conj(D[a, c, d, b])
            @tensor E2[a, b, c, d] := ROCArray(C)[a, b, c, d] +
                β * conj(ROCArray(D)[a, c, d, b])
            @test collect(E2) ≈ E1
        end

        @testset "tensortrace" begin
            A = randn(Float32, (5, 10, 10))
            @tensor B1[a] := A[a, b′, b′]
            @tensor B2[a] := ROCArray(A)[a, b′, b′]
            @test collect(B2) ≈ B1

            C = randn(ComplexF32, (3, 20, 5, 3, 20, 4, 5))
            @tensor D1[e, a, d] := C[a, b, c, d, b, e, c]
            @tensor D2[e, a, d] := ROCArray(C)[a, b, c, d, b, e, c]
            @test collect(D2) ≈ D1

            @tensor D3[a, e, d] := conj(C[a, b, c, d, b, e, c])
            @tensor D4[a, e, d] := conj(ROCArray(C)[a, b, c, d, b, e, c])
            @test collect(D4) ≈ D3

            α = randn(ComplexF32)
            @tensor D5[d, e, a] := α * C[a, b, c, d, b, e, c]
            @tensor D6[d, e, a] := α * ROCArray(C)[a, b, c, d, b, e, c]
            @test collect(D6) ≈ D5
        end

        @testset "tensorcontract" begin
            A = randn(Float32, (3, 20, 5, 3, 4))
            B = randn(Float32, (5, 6, 20, 3))
            @tensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
            @tensor C2[a, g, e, d, f] := ROCArray(A)[a, b, c, d, e] * ROCArray(B)[c, f, b, g]
            @test collect(C2) ≈ C1

            D = randn(ComplexF32, (5, 5, 5))
            E = rand(ComplexF32, (5, 5, 5))
            @tensor F1[a, b, c, d, e, f] := D[a, b, c] * conj(E[d, e, f])
            @tensor F2[a, b, c, d, e, f] := ROCArray(D)[a, b, c] * conj(ROCArray(E)[d, e, f])
            @test collect(F2) ≈ F1
            D = randn(Float64, (5, 5, 5))
            E = rand(ComplexF64, (5, 5, 5))
            @tensor F1[a, b, c, d, e, f] := D[a, b, c] * conj(E[d, e, f])
            @tensor F2[a, b, c, d, e, f] := ROCArray(D)[a, b, c] * conj(ROCArray(E)[d, e, f])
            @test collect(F2) ≈ F1
        end
    end

    @testset "more complicated expressions" verbose = true begin
        Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
        A = rand(Float32, (Dc, Da, Df, Da, De, Db, Db, Dg))
        B = rand(Float32, (Dc, Dh, Dg, De, Dd))
        C = rand(Float32, (Dd, Dh, Df))

        @tensor D1[d, f, h] := A[c, a, f, a, e, b, b, g] * B[c, h, g, e, d] +
            Float32(0.5) * C[d, h, f]
        @tensor D2[d, f, h] := ROCArray(A)[c, a, f, a, e, b, b, g] *
            ROCArray(B)[c, h, g, e, d] + Float32(0.5) * ROCArray(C)[d, h, f]
        @test collect(D2) ≈ D1

        @test norm(vec(D1)) ≈ sqrt(abs(@tensor D1[d, f, h] * conj(D1[d, f, h])))
        @test norm(D2) ≈ sqrt(abs(@tensor D2[d, f, h] * conj(D2[d, f, h])))

        @testset "readme example" begin
            α = randn()
            A = randn(5, 5, 5, 5, 5, 5)
            B = randn(5, 5, 5)
            C = randn(5, 5, 5)
            D = zeros(5, 5, 5)
            D2 = ROCArray(D)
            @tensor begin
                D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
                E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
            end
            @tensor begin
                D2[a, b, c] = ROCArray(A)[a, e, f, c, f, g] * ROCArray(B)[g, b, e] +
                    α * ROCArray(C)[c, a, b]
                E2[a, b, c] := ROCArray(A)[a, e, f, c, f, g] * ROCArray(B)[g, b, e] +
                    α * ROCArray(C)[c, a, b]
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
                HRAA2[a, s1, s2, c] := ROCArray(ρₗ)[a, a'] * ROCArray(A1)[a', t1, b] *
                    ROCArray(A2)[b, t2, c'] * ROCArray(ρᵣ)[c', c] * ROCArray(H)[s1, s2, t1, t2]
            end
            @test HRAA2 isa ROCArray{T}
            @test collect(HRAA2) ≈ HRAA1

            allocator = AMDAllocator()
            @tensor backend = hipTENSORBackend() allocator = allocator begin
                HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @test HRAA3 isa ROCArray{T, 4}
            @test collect(HRAA3) ≈ HRAA1

            @tensor begin
                E1 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] * H[t, t', s, s'] *
                    conj(A1[a', t, b']) * conj(A2[b', t', c'])
                E2 = ROCArray(ρₗ)[a', a] * ROCArray(A1)[a, s, b] * ROCArray(A2)[b, s', c] *
                    ROCArray(ρᵣ)[c, c'] * ROCArray(H)[t, t', s, s'] *
                    conj(ROCArray(A1)[a', t, b']) * conj(ROCArray(A2)[b', t', c'])
            end
            @tensor backend = hipTENSORBackend() allocator = AMDAllocator() begin
                E3 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] * H[t, t', s, s'] *
                    conj(A1[a', t, b']) * conj(A2[b', t', c'])
            end
            @test E1 ≈ E2 ≈ E3
        end
    end

    @testset "BufferAllocator" verbose = true begin
        # is the memory of `A` taken from `buffer`?
        function isbufferbacked(A, buffer)
            iszero(length(buffer)) && return false
            base = UInt(pointer(buffer))
            return base ≤ UInt(pointer(A)) < base + length(buffer)
        end

        @testset "Constructor and basic properties" begin
            buffer = AMDBufferAllocator(; sizehint = 1024)
            @test buffer isa BufferAllocator{ROCArray{UInt8, 1, AMDGPU.Mem.HIPBuffer}}
            @test length(buffer) == 1024
            @test isempty(buffer)
            @test buffer.offset == 0

            # resizing frees the old buffer and allocates a new one
            resize!(buffer, 3000)
            @test length(buffer) == 4096
            @test isempty(buffer)
            buffer.offset = 100
            @test_throws ErrorException resize!(buffer, 8192)
            empty!(buffer)
            @test length(resize!(buffer, 8192)) == 8192
        end

        @testset "tensoralloc" begin
            buffer = AMDBufferAllocator(; sizehint = 4096)

            # temporaries are taken from the buffer, in the memory space of the buffer
            C1 = tensoralloc(ROCArray{Float32, 2}, (8, 8), Val(true), buffer)
            @test C1 isa ROCArray{Float32, 2, AMDGPU.Mem.HIPBuffer}
            @test size(C1) == (8, 8)
            @test isbufferbacked(C1, buffer)
            @test buffer.offset == 8 * 8 * sizeof(Float32)

            # non-temporaries are not
            offset = buffer.offset
            C2 = tensoralloc(ROCArray{Float32, 2}, (8, 8), Val(false), buffer)
            @test C2 isa ROCArray{Float32, 2}
            @test !isbufferbacked(C2, buffer)
            @test buffer.offset == offset

            # freeing a buffer-backed tensor does not invalidate the buffer
            ptr1 = pointer(C1)
            tensorfree!(C1, buffer)
            allocator_reset!(buffer, 0)
            C3 = tensoralloc(ROCArray{Float32, 2}, (8, 8), Val(true), buffer)
            @test isbufferbacked(C3, buffer)
            @test pointer(C3) == ptr1
            fill!(C3, 1.0f0)
            @test all(isone, collect(C3))

        end

        @testset "storage mismatch falls back" begin
            # a host buffer cannot back ROCArrays
            hostbuffer = BufferAllocator(; sizehint = 4096)
            C1 = tensoralloc(ROCArray{Float32, 2}, (8, 8), Val(true), hostbuffer)
            @test C1 isa ROCArray{Float32, 2}
            @test hostbuffer.offset == 0

            # a device buffer cannot back Arrays
            cubuffer = AMDBufferAllocator(; sizehint = 4096)
            C2 = tensoralloc(Array{Float64, 2}, (8, 8), Val(true), cubuffer)
            @test C2 isa Array{Float64, 2}
            @test cubuffer.offset == 0
        end

        @testset "alignment" begin
            # cuTENSOR is sensitive to this: it only selects its fastest kernels for
            # 256-byte aligned data, so every temporary has to be padded to that
            buffer = AMDBufferAllocator(; sizehint = 8192)
            @test iszero(UInt(pointer(buffer)) % 256)

            # a deliberately misaligning allocation of 3 bytes
            C1 = tensoralloc(ROCArray{UInt8, 1}, (3,), Val(true), buffer)
            @test isbufferbacked(C1, buffer)
            @test buffer.offset == 3
            for T in (Float32, Float64, ComplexF32, ComplexF64)
                C2 = tensoralloc(ROCArray{T, 1}, (4,), Val(true), buffer)
                @test isbufferbacked(C2, buffer)
                @test iszero(UInt(pointer(C2)) % 256)
            end

            # host buffers stay at the alignment Julia actually guarantees
            @test TensorOperations.buffer_alignment(BufferAllocator()) == 16
        end

        @testset "checkpoint and reset" begin
            buffer = AMDBufferAllocator(; sizehint = 4096)
            cp0 = allocator_checkpoint!(buffer)
            @test cp0 == 0

            C1 = tensoralloc(ROCArray{Float32, 2}, (8, 8), Val(true), buffer)
            cp1 = allocator_checkpoint!(buffer)
            @test cp1 > cp0
            C2 = tensoralloc(ROCArray{Float32, 2}, (8, 8), Val(true), buffer)
            @test pointer(C2) != pointer(C1)

            allocator_reset!(buffer, cp1)
            @test buffer.offset == cp1
            @test_throws ArgumentError allocator_reset!(buffer, cp1 + 10)

            allocator_reset!(buffer, cp0)
            @test isempty(buffer)
        end

        @testset "tensor network ($T)" for T in
            (Float32, Float64, ComplexF32, ComplexF64)
            D1, D2, D3 = 30, 40, 20
            d1, d2 = 2, 3

            A1 = ROCArray(randn(T, D1, d1, D2))
            A2 = ROCArray(randn(T, D2, d2, D3))
            ρₗ = ROCArray(randn(T, D1, D1))
            ρᵣ = ROCArray(randn(T, D3, D3))
            H = ROCArray(randn(T, d1, d2, d1, d2))

            @tensor begin
                HRAA1[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end

            buffer = AMDBufferAllocator()
            @tensor allocator = buffer begin
                HRAA2[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @test HRAA2 isa ROCArray{T, 4}
            @test collect(HRAA2) ≈ collect(HRAA1)

            # all temporaries were reclaimed, and the buffer was actually used
            @test buffer.offset == 0
            @test buffer.max_offset > 0

            # The high-water mark only counts the temporaries that actually fit in the
            # buffer, so it may still grow while the buffer is warming up, but it has to
            # converge to a fixed size after a couple of contractions.
            max0 = buffer.max_offset
            for _ in 1:5
                @tensor allocator = buffer begin
                    HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                        ρᵣ[c', c] * H[s1, s2, t1, t2]
                end
                @test collect(HRAA3) ≈ collect(HRAA1)
            end
            max1 = buffer.max_offset
            @test max1 ≥ max0
            @test length(buffer) ≥ max1

            for _ in 1:5
                @tensor allocator = buffer begin
                    HRAA3[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                        ρᵣ[c', c] * H[s1, s2, t1, t2]
                end
                @test collect(HRAA3) ≈ collect(HRAA1)
            end
            @test buffer.offset == 0
            @test buffer.max_offset == max1

            # scalar output
            @tensor begin
                E1 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] *
                    H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c'])
            end
            @tensor allocator = buffer begin
                E2 = ρₗ[a', a] * A1[a, s, b] * A2[b, s', c] * ρᵣ[c, c'] *
                    H[t, t', s, s'] * conj(A1[a', t, b']) * conj(A2[b', t', c'])
            end
            @test E1 ≈ E2
            @test buffer.offset == 0
        end

        @testset "host inputs are promoted to ROCArray" begin
            D1, D2, D3 = 30, 40, 20
            d1, d2 = 2, 3
            T = Float64

            A1 = randn(T, D1, d1, D2)
            A2 = randn(T, D2, d2, D3)
            ρₗ = randn(T, D1, D1)
            ρᵣ = randn(T, D3, D3)
            H = randn(T, d1, d2, d1, d2)

            @tensor begin
                HRAA1[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end

            buffer = AMDBufferAllocator()
            @tensor backend = hipTENSORBackend() allocator = buffer begin
                HRAA2[a, s1, s2, c] := ρₗ[a, a'] * A1[a', t1, b] * A2[b, t2, c'] *
                    ρᵣ[c', c] * H[s1, s2, t1, t2]
            end
            @test HRAA2 isa ROCArray{T, 4}
            @test collect(HRAA2) ≈ HRAA1
            @test buffer.offset == 0
            @test buffer.max_offset > 0
        end

        @testset "ncon" begin
            A = ROCArray(randn(Float32, 5, 5))
            B = ROCArray(randn(Float32, 5, 5))
            C = ROCArray(randn(Float32, 5, 5))
            buffer = AMDBufferAllocator()

            R = ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
            @test R isa ROCArray{Float32, 2}
            @test collect(R) ≈ collect(A) * collect(B) * collect(C)
            @test buffer.offset == 0

            max1 = buffer.max_offset
            for _ in 1:5
                ncon([A, B, C], [[-1, 1], [1, 2], [2, -2]]; allocator = buffer)
            end
            @test buffer.offset == 0
            @test buffer.max_offset == max1
        end
    end

    @testset "@hiptensor" verbose = true begin
        @testset "tensorcontract 1" begin
            A = randn(Float64, (3, 5, 4, 6))
            @tensor C1[4, 1, 3, 2] := A[1, 2, 3, 4]
            @hiptensor C2[4, 1, 3, 2] := A[1, 2, 3, 4]
            @test C1 ≈ collect(C2)
            @test_throws IndexError begin
                @hiptensor C[1, 2, 3, 4] := A[1, 2, 3]
            end
            @test_throws IndexError begin
                @hiptensor C[1, 2, 3, 4] := A[1, 2, 2, 4]
            end

            B = randn(Float64, (5, 6, 3, 4))
            p = [3, 1, 4, 2]
            @tensor C1[3, 1, 4, 2] := A[3, 1, 4, 2] + B[1, 2, 3, 4]
            @hiptensor C2[3, 1, 4, 2] := A[3, 1, 4, 2] + B[1, 2, 3, 4]
            @test C1 ≈ collect(C2)
            @test_throws DimensionMismatch begin
                @hiptensor C[1, 2, 3, 4] := A[1, 2, 3, 4] + B[1, 2, 3, 4]
            end

            A = randn(Float64, (50, 100, 100))
            @tensor C1[a] := A[a, b', b']
            @hiptensor C2[a] := A[a, b', b']
            @test C1 ≈ collect(C2)

            A = randn(Float64, (3, 20, 5, 3, 20, 4, 5))
            @tensor C1[e, a, d] := A[a, b, c, d, b, e, c]
            @hiptensor C2[e, a, d] := A[a, b, c, d, b, e, c]
            @test C1 ≈ collect(C2)

            A = randn(Float64, (3, 20, 5, 3, 4))
            B = randn(Float64, (5, 6, 20, 3))
            @tensor C1[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
            @hiptensor C2[a, g, e, d, f] := A[a, b, c, d, e] * B[c, f, b, g]
            @test C1 ≈ collect(C2)
            @test_throws IndexError begin
                @hiptensor A[a, b, c, d] * B[c, f, b, g]
            end
        end

        @testset "tensorcontract 2" begin
            A = randn(Float64, (5, 5, 5, 5))
            B = rand(ComplexF64, (5, 5, 5, 5))
            @tensor C1[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
            @hiptensor C2[1, 2, 5, 6, 3, 4, 7, 8] := A[1, 2, 3, 4] * B[5, 6, 7, 8]
            @test C1 ≈ collect(C2)
            @test_throws IndexError begin
                @hiptensor C[a, b, c, d, e, f, g, i] := A[a, b, c, d] * B[e, f, g, h]
            end
        end

        @testset "tensorcontract 3" begin
            Da, Db, Dc, Dd, De, Df, Dg, Dh = 10, 15, 4, 8, 6, 7, 3, 2
            A = rand(ComplexF64, (Da, Dc, Df, Da, De, Db, Db, Dg))
            B = rand(ComplexF64, (Dc, Dh, Dg, De, Dd))
            C = rand(ComplexF64, (Dd, Dh, Df))
            @tensor D1[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] +
                0.5 * C[d, h, f]
            @hiptensor D2[d, f, h] := A[a, c, f, a, e, b, b, g] * B[c, h, g, e, d] +
                0.5 * C[d, h, f]
            @test D1 ≈ collect(D2)
            E1 = sqrt(abs((@tensor tensorscalar(D1[d, f, h] * conj(D1[d, f, h])))))
            E2 = sqrt(abs((@hiptensor tensorscalar(D2[d, f, h] * conj(D2[d, f, h])))))
            @test E1 ≈ E2
        end

        @testset "views" begin
            p = [3, 1, 4, 2]
            Abig = AMDGPU.randn(Float32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8)
            )
            Cbig = AMDGPU.zeros(Float32, (50, 50, 50, 50))
            C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
            Acopy = copy(A)
            Ccopy = copy(C)
            @tensor C[3, 1, 4, 2] = A[1, 2, 3, 4]
            @tensor Ccopy[3, 1, 4, 2] = Acopy[1, 2, 3, 4]
            @test copy(C) ≈ Ccopy
            @test_throws TensorOperations.IndexError begin
                @tensor C[3, 1, 4, 2] = A[1, 2, 3]
            end
            @test_throws DimensionMismatch begin
                @tensor C[3, 1, 4, 2] = A[3, 1, 4, 2]
            end
            @test_throws TensorOperations.IndexError begin
                @tensor C[1, 1, 2, 3] = A[1, 2, 3, 4]
            end
        end

        @testset "views 2" begin
            p = [3, 1, 4, 2]
            Abig = AMDGPU.randn(Float32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:9), 2 .+ 2 .* (0:6), 5 .+ 4 .* (0:6), 4 .+ 3 .* (0:8)
            )
            Cbig = AMDGPU.zeros(Float32, (50, 50, 50, 50))
            C = view(Cbig, 13 .+ (0:6), 11 .+ 4 .* (0:9), 15 .+ 4 .* (0:8), 4 .+ 3 .* (0:6))
            Acopy = permutedims(copy(A), p)
            Ccopy = copy(C)
            α = randn(Float32)
            β = randn(Float32)
            @tensor C[3, 1, 4, 2] = β * C[3, 1, 4, 2] + α * A[1, 2, 3, 4]
            Ccopy = β * Ccopy + α * Acopy
            @test copy(C) ≈ Ccopy
            @test_throws ArgumentError @macroexpand(
                @tensor C[3, 1, 4, 2] = 0.5 * C[3, 1, 4, 2] + 1.2 * A[1, 2, 3]
            )
            @test_throws DimensionMismatch begin
                @tensor C[3, 1, 4, 2] = 0.5 * C[3, 1, 4, 2] + 1.2 * A[3, 1, 2, 4]
            end
            @test_throws ArgumentError @macroexpand(
                @tensor C[1, 1, 2, 3] = 0.5 * C[1, 1, 2, 3] + 1.2 * A[1, 2, 3, 4]
            )
        end

        @testset "views 3" begin
            Abig = AMDGPU.rand(Float32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6), 7 .+ 2 .* (0:8)
            )
            Bbig = AMDGPU.rand(Float32, (50, 50))
            B = view(Bbig, 13 .+ (0:14), 3 .+ 5 .* (0:6))
            Acopy = copy(A)
            Bcopy = copy(B)
            α = randn(Float32)
            @tensor B[b, c] += α * A[a, b, c, a]
            @tensor Bcopy[b, c] += α * Acopy[a, b, c, a]
            @test copy(B) ≈ Bcopy
            @test_throws IndexError begin
                @tensor B[b, c] += α * A[a, b, c]
            end
            @test_throws DimensionMismatch begin
                @tensor B[c, b] += α * A[a, b, c, a]
            end
            @test_throws ArgumentError @macroexpand(@tensor B[c, b] += α * A[a, b, a, a])
            @test_throws DimensionMismatch begin
                @tensor B[c, b] += α * A[a, b, a, c]
            end
        end

        @testset "views 4" begin
            Abig = AMDGPU.rand(Float32, (30, 30, 30, 30))
            A = view(
                Abig, 1 .+ 3 .* (0:8), 2 .+ 2 .* (0:14), 5 .+ 4 .* (0:6),
                7 .+ 2 .* (0:8)
            )
            Bbig = AMDGPU.rand(Float32, (50, 50, 50))
            B = view(Bbig, 3 .+ 5 .* (0:6), 7 .+ 2 .* (0:7), 13 .+ (0:14))
            Cbig = AMDGPU.rand(Float32, (40, 40, 40))
            C = view(Cbig, 3 .+ 2 .* (0:8), 13 .+ (0:8), 7 .+ 3 .* (0:7))
            Acopy = copy(A)
            Bcopy = copy(B)
            Ccopy = copy(C)
            α = randn(Float32)
            @tensor C[d, a, e] -= α * A[a, b, c, d] * conj(B[c, e, b])
            @tensor Ccopy[d, a, e] -= α * Acopy[a, b, c, d] * conj(Bcopy[c, e, b])
            @test copy(C) ≈ Ccopy
        end
    end

    @testset "unsupported types fall back to Strided" begin
        using TensorOperations: select_backend, StridedNative, StridedBLAS
        using Strided: StridedView
        sv(T, dims...) = StridedView(ROCArray(rand(T, dims...)))

        A32, C32 = sv(Float32, 4, 4), sv(Float32, 4, 4)
        @test select_backend(tensoradd!, C32, A32) == hipTENSORBackend()
        @test select_backend(tensortrace!, sv(Float32, 4), sv(Float32, 4, 4, 4)) ==
            hipTENSORBackend()
        @test select_backend(tensorcontract!, C32, A32, A32) == hipTENSORBackend()

        Az, Cz = sv(ComplexF32, 4, 4), sv(ComplexF32, 4, 4)
        @test select_backend(tensoradd!, Cz, Az) == StridedNative()
        @test select_backend(tensortrace!, sv(ComplexF32, 4), sv(ComplexF32, 4, 4, 4)) ==
            StridedNative()
        @test select_backend(tensorcontract!, Cz, Az, Az) == hipTENSORBackend()
        @test select_backend(tensorcontract!, Cz, A32, Az) == StridedBLAS()
        Ai = sv(Int, 4, 4)
        @test select_backend(tensorcontract!, Ai, Ai, Ai) == StridedNative()

        # results stay on the device and match the host
        A = randn(ComplexF64, (3, 4, 5))
        @tensor C1[c, a, b] := 2 * conj(A[a, b, c])
        C2 = similar(ROCArray(A), (5, 3, 4))
        @tensor backend = hipTENSORBackend() C2[c, a, b] = 2 * conj(ROCArray(A)[a, b, c])
        @test C2 isa ROCArray
        @test collect(C2) ≈ C1

        A = rand(-5:5, (3, 4))
        B = rand(-5:5, (4, 5))
        @tensor C1[a, c] := A[a, b] * B[b, c]
        @tensor C2[a, c] := ROCArray(A)[a, b] * ROCArray(B)[b, c]
        @test C2 isa ROCArray
        @test collect(C2) == C1
    end

    @testset "tensortrace! does not leak plans" begin
        # each leaked plan holds on to a 128 KiB reduction workspace
        A = ROCArray(randn(Float32, 64, 64, 64, 64))
        C = AMDGPU.zeros(Float32, 64, 64)
        @tensor C[a, b] = A[a, c, b, c] # warm up

        GC.gc(true)
        AMDGPU.reclaim()
        AMDGPU.synchronize()
        GC.enable(false)
        try
            live0 = AMDGPU.memory_stats().live
            for _ in 1:1000
                @tensor C[a, b] = A[a, c, b, c]
            end
            AMDGPU.synchronize()
            @test AMDGPU.memory_stats().live - live0 < 2^20 # would be 125 MiB if leaking
        finally
            GC.enable(true)
        end
    end

    @testset "Issues" verbose = true begin
        @testset "Issue PR #186" begin
            # https://github.com/Jutho/TensorOperations.jl/pull/186
            A = randn(Float32, (5, 5, 5, 5))
            Atr = @tensor A[a, b, b, a]
            Atr2 = @hiptensor A[a, b, b, a]
            @test Atr ≈ Atr2
        end
    end
end
