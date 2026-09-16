# Flattens a benchmark result into a DataFrame, joining timings back against specs
# (regenerated from REGISTRY) for GFLOP/s and bandwidth.

"""
    resultstable(results::BenchmarkGroup; categories=collect(keys(REGISTRY)), sizes=nothing)

Flatten a benchmark-run result (with the same `[category][provider][id]["benchmark"]` nesting
`build_suite` produces) into a `DataFrame` with columns `category`, `provider`, `id`, `params`
(the originating `NamedTuple`), `mintime` (ns), `allocs`, `memory` (bytes), `gflops`
(`flops(spec) / mintime`, or `missing` if `mintime` is zero), and `gbps` (`bytes(spec) /
mintime`). `categories`/`sizes` must match what was passed to the `build_suite` call that
produced `results`, since specs (and therefore flop/byte counts) are regenerated from
`REGISTRY` rather than stored in the result itself.
"""
function resultstable(
        results::BenchmarkGroup; categories = collect(keys(REGISTRY)), sizes = nothing
    )
    rows = NamedTuple[]
    for category in categories
        haskey(results, String(category)) || continue
        cases = REGISTRY[category](_sizes_for(sizes, category))
        casesbyid = Dict(c.id => c for c in cases)
        catgroup = results[String(category)]
        for providerlabel in keys(catgroup)
            provgroup = catgroup[providerlabel]
            for id in keys(provgroup)
                trial = provgroup[id]["benchmark"]
                case = casesbyid[id]
                mintime = minimum(trial.times)
                fl = flops(case.spec)
                by = bytes(case.spec)
                gflops = mintime > 0 ? fl / mintime : missing
                gbps = mintime > 0 ? by / mintime : missing
                push!(
                    rows,
                    (;
                        category = String(category), provider = providerlabel, id,
                        params = case.params, mintime, allocs = trial.allocs,
                        memory = trial.memory, gflops, gbps,
                    )
                )
            end
        end
    end
    return DataFrame(rows)
end
