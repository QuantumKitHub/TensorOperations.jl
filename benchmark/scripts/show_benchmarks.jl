#!/usr/bin/env julia
# Plots time/GFLOPs-vs-size curves from a run_benchmarks.jl result JSON. Requires CairoMakie
# (`Pkg.add("CairoMakie")` into this environment first -- not a package dependency, it's heavy).
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using PkgBenchmark
using CairoMakie
using TensorOperationsBenchmarks

function parse_commandline(args)
    s = ArgParseSettings(; description = "Plot GFLOP/s-vs-size scaling curves from a PkgBenchmark result.")
    @add_arg_table! s begin
        "resultfile"
        help = "path to a PkgBenchmark result JSON, as written by run_benchmarks.jl"
        required = true
        "--out"
        help = "output image path (default: replace the input's extension with .png)"
        default = nothing
    end
    return parse_args(args, s)
end

function main(args)
    opts = parse_commandline(args)
    results = PkgBenchmark.readresults(opts["resultfile"])
    group = PkgBenchmark.benchmarkgroup(results)

    rows = resultstable(group)

    fig = Figure(; size = (1000, 800))
    categories = unique(r.category for r in rows)
    for (i, category) in enumerate(categories)
        ax = Axis(
            fig[fldmod1(i, 2)...]; xscale = log2, yscale = log10,
            title = category, xlabel = "size", ylabel = "GFLOP/s"
        )
        catrows = filter(r -> r.category == category, rows)
        for provider in unique(r.provider for r in catrows)
            provrows = filter(r -> r.provider == provider, catrows)
            sort!(provrows; by = r -> get(r.params, :dim, get(r.params, :D, 0)))
            xs = [get(r.params, :dim, get(r.params, :D, 0)) for r in provrows]
            ys = [r.gflops for r in provrows]
            lines!(ax, xs, ys; label = provider)
            scatter!(ax, xs, ys)
        end
        axislegend(ax)
    end

    outfile = something(opts["out"], splitext(opts["resultfile"])[1] * ".png")
    save(outfile, fig)
    @info "Wrote $outfile"
    return 0
end

@main
