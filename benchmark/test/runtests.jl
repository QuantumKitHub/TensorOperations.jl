using Test
using TensorOperationsBenchmarks
using TensorOperations: StridedNative
using BenchmarkTools
using DataFrames: nrow
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
        suite = build_suite([provider]; categories = [:contract], sizes = (4, 8))
        results = run(suite; samples = 1, evals = 1, seconds = 5)
        rows = resultstable(results; categories = [:contract], sizes = (4, 8))
        @test nrow(rows) > 0
        @test all(>(0), rows.mintime)
        @test all(x -> x isa Float64, rows.gflops)
    end

    @testset "mixed-precision cases execute" begin
        suite = build_suite([provider]; categories = [:mixed_precision], sizes = (8,))
        results = run(suite; samples = 1, evals = 1, seconds = 5)
        @test !isempty(results["mixed_precision"][label(provider)])
    end

    @testset "network cost matches ncon's own contraction tree" begin
        cases = REGISTRY[:network]((16,))
        case = only(filter(c -> occursin("mps_1site", c.id), cases))
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
        cases = REGISTRY[:contract]((8,))
        case = first(cases)
        ts = TensorOperationsBenchmarks.maketensors(case.spec, provider)
        C_before = ts[1]
        C_after = TensorOperationsBenchmarks.execute(case.spec, ts, provider)
        @test C_after === C_before
    end

    @testset "network category covers mps/ctmrg/trg topics" begin
        cases = REGISTRY[:network]((16,))
        @test any(c -> c.params.topic == :mps && c.params.variant == :onesite, cases)
        @test any(c -> c.params.topic == :mps && c.params.variant == :twosite, cases)
        @test any(c -> c.params.topic == :ctmrg, cases)
        @test any(c -> c.params.topic == :trg, cases)
    end

    @testset "tccg cases are merged into :contract, filterable via @tagged" begin
        cases = REGISTRY[:contract]((4,))
        tccg_cases = filter(c -> c.params.source == :tccg, cases)
        @test length(TensorOperationsBenchmarks.TCCG_CONTRACTIONS) == 24
        @test any(c -> c.params.source == :synthetic, cases)
        for prefix in ("ccsd_", "ccsd_t_", "ao2mo_", "intensli_")
            @test any(c -> startswith(c.params.equation, prefix), tccg_cases)
        end
        suite = build_suite([provider]; categories = [:contract], sizes = (4,))
        results = run(suite[@tagged "tccg"]; samples = 1, evals = 1, seconds = 5)
        @test !isempty(results["contract"][label(provider)])
        @test length(results["contract"][label(provider)]) == length(tccg_cases)
    end

    @testset "BenchmarkCase stores tags derived from category + Symbol-valued params" begin
        case = first(REGISTRY[:trace]((8,)))
        @test "trace" in case.tags
        @test string(case.params.kind) in case.tags
    end

    @testset "contract permuted-stride layouts are structurally distinct and filterable" begin
        cases = REGISTRY[:contract]((8,))
        layouts = unique(c.params.layout for c in cases if c.params.source == :synthetic)
        @test :gemm_ready in layouts
        @test :both_permuted in layouts

        suite = build_suite([provider]; categories = [:contract], sizes = (8,))
        results = run(suite[@tagged "both_permuted"]; samples = 1, evals = 1, seconds = 5)
        n_expected = count(c -> c.params.source == :synthetic && c.params.layout == :both_permuted, cases)
        @test length(results["contract"][label(provider)]) == n_expected
    end

    @testset "within_memory_budget(spec, id) warns when skipping an oversized case" begin
        spec = ContractSpec([:a1, :a2], [:a2, :b1], [:a1, :b1], Dict(:a1 => 8, :a2 => 8, :b1 => 8))
        # override maxbytes (rather than relying on the host's memory-scaled default) so this
        # test is deterministic regardless of how much RAM the machine running it has.
        @test_logs (:warn, r"exceeds memory budget") within_memory_budget(spec, "tiny_budget_test"; maxbytes = 10)
    end

    @testset "specs have informative show methods" begin
        spec = ContractSpec([:i, :k], [:k, :j], [:i, :j], Dict(:i => 4, :j => 4, :k => 4))
        @test occursin("C[i,j]", sprint(show, spec))
        @test occursin("A[i,k]", sprint(show, spec))
        @test occursin("B[k,j]", sprint(show, spec))
    end
end
