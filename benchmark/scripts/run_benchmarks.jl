#!/usr/bin/env julia
# CLI wrapper around PkgBenchmark.benchmarkpkg. `--threads` relaunches Julia per value (fixed
# at startup); `--blas-threads`/`--strided-threads` set env vars benchmarks.jl reads.
#   julia --project=. scripts/run_benchmarks.jl --threads 1 2 4 --blas-threads 1 4 --out results
using Pkg
Pkg.activate(@__DIR__ * "/..")

using ArgParse
using PkgBenchmark

function parse_commandline()
    s = ArgParseSettings(; description = "Run the TensorOperationsBenchmarks suite via PkgBenchmark.")
    @add_arg_table! s begin
        "--threads"
        help = "outer Julia process thread count(s) to sweep (relaunches Julia per value)"
        arg_type = Int
        nargs = '*'
        default = [Threads.nthreads()]
        "--blas-threads"
        help = "inner BLAS thread count(s) to sweep"
        arg_type = Int
        nargs = '*'
        default = Int[]
        "--strided-threads"
        help = "inner Strided.jl thread count(s) to sweep"
        arg_type = Int
        nargs = '*'
        default = Int[]
        "--out"
        help = "output file prefix (a suffix identifying the thread combo and `.json` are appended)"
        default = "results"
    end
    return parse_args(s)
end

opts = parse_commandline()
blascounts = isempty(opts["blas-threads"]) ? [nothing] : opts["blas-threads"]
stridedcounts = isempty(opts["strided-threads"]) ? [nothing] : opts["strided-threads"]

for nthreads in opts["threads"], blas in blascounts, strided in stridedcounts
    @info "Running benchmarks" nthreads blas strided
    withenv(
        "TOB_BLAS_THREADS" => blas === nothing ? "" : string(blas),
        "TOB_STRIDED_THREADS" => strided === nothing ? "" : string(strided),
    ) do
        cfg = BenchmarkConfig(; juliacmd = `julia -t $nthreads -O3`)
        results = benchmarkpkg(dirname(@__DIR__), cfg)
        outfile = "$(opts["out"])_t$(nthreads)_blas$(blas)_strided$(strided).json"
        writeresults(outfile, results)
        @info "Wrote $outfile"
    end
end
