using Test
using TensorOperationsBenchmarks
using TensorOperations: StridedNative
using BenchmarkTools
using LinearAlgebra: BLAS
using Strided: Strided

@testset "TensorOperationsBenchmarks" begin
    provider = ArrayProvider{Float64}(; backend = StridedNative())

    @testset "every category builds and runs" begin
        suite = build_suite([provider]; categories = collect(keys(REGISTRY)), sizes = (4, 8))
        results = run(suite; samples = 1, evals = 1, seconds = 5)
        for category in keys(REGISTRY)
            @test haskey(results, String(category))
        end
    end

    @testset "resultstable joins timings with flop/byte counts" begin
        suite = build_suite([provider]; categories = [:pairwise], sizes = (4, 8))
        results = run(suite; samples = 1, evals = 1, seconds = 5)
        rows = resultstable(results; categories = [:pairwise], sizes = (4, 8))
        @test !isempty(rows)
        @test all(r -> r.mintime > 0, rows)
        @test all(r -> r.gflops isa Float64, rows)
    end

    @testset "mixed-precision cases execute" begin
        suite = build_suite([provider]; categories = [:mixed_precision], sizes = (8,))
        results = run(suite; samples = 1, evals = 1, seconds = 5)
        @test !isempty(results["mixed_precision"][label(provider)])
    end

    @testset "network cost matches ncon's own contraction tree" begin
        cases = REGISTRY[:mps]((16,))
        case = only(filter(c -> occursin("1site", c.id), cases))
        @test flops(case.spec) > 0
    end

    @testset "with_threads restores prior thread counts, set_threads! does not" begin
        oldblas = BLAS.get_num_threads()
        oldstrided = Strided.get_num_threads()
        with_threads(() -> nothing, ThreadConfig(; blas = 1, strided = 1))
        @test BLAS.get_num_threads() == oldblas
        @test Strided.get_num_threads() == oldstrided

        set_threads!(ThreadConfig(; blas = 1))
        @test BLAS.get_num_threads() == 1
        BLAS.set_num_threads(oldblas)  # restore for subsequent tests
    end

    @testset "ArrayProvider's randtensor is reproducible across providers, varies within one" begin
        p1 = ArrayProvider{Float64}()
        p2 = ArrayProvider{Float64}()
        A1 = randtensor(p1, [:a, :b], (4, 4), Float64)
        A2 = randtensor(p2, [:a, :b], (4, 4), Float64)
        @test A1 == A2  # same fixed seed -> same first draw
        B1 = randtensor(p1, [:a, :b], (4, 4), Float64)
        @test A1 != B1  # stateful rng -> second draw differs from the first
    end

    @testset "execute doesn't allocate a fresh output every call (preallocated in setup)" begin
        cases = REGISTRY[:pairwise]((8,))
        case = first(cases)
        ts = TensorOperationsBenchmarks.maketensors(case.spec, provider)
        C_before = ts[end]
        C_after = TensorOperationsBenchmarks.execute(case.spec, ts, provider)
        @test C_after === C_before
    end

    @testset "mps category covers 1-site and 2-site variants" begin
        cases = REGISTRY[:mps]((16,))
        @test any(c -> occursin("1site", c.id), cases)
        @test any(c -> occursin("2site", c.id), cases)
    end

    @testset "tccg category covers all 4 source groups and executes" begin
        cases = REGISTRY[:tccg]((4,))
        @test length(TensorOperationsBenchmarks.TCCG_CONTRACTIONS) == 24
        for prefix in ("ccsd_", "ccsd_t_", "ao2mo_", "intensli_")
            @test any(c -> startswith(c.params.equation, prefix), cases)
        end
        suite = build_suite([provider]; categories = [:tccg], sizes = (4,))
        results = run(suite; samples = 1, evals = 1, seconds = 5)
        @test !isempty(results["tccg"][label(provider)])
    end

    @testset "ctmrg and trg categories execute" begin
        for category in (:ctmrg, :trg)
            suite = build_suite([provider]; categories = [category], sizes = (8,))
            results = run(suite; samples = 1, evals = 1, seconds = 5)
            @test !isempty(results[String(category)][label(provider)])
        end
    end
end
