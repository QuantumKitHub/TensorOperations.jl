# Flattens a benchmark result into rows, joining timings back against specs (regenerated from
# REGISTRY) for GFLOP/s and bandwidth.

"""
    ResultRow

One row of [`resultstable`](@ref): `category`, `provider`, `id`, `params`, `mintime` (ns),
`allocs`, `memory` (bytes), `gflops` (`flops(spec) / mintime`, or `missing` if `mintime` is
zero), and `gbps` (`bytes(spec) / mintime`).
"""
struct ResultRow
    category::String
    provider::String
    id::String
    params::NamedTuple
    mintime::Float64
    allocs::Int
    memory::Int
    gflops::Union{Float64, Missing}
    gbps::Union{Float64, Missing}
end

"""
    resultstable(results::BenchmarkGroup; categories=collect(keys(REGISTRY)), sizes=nothing)

Flatten a benchmark-run result (with the same `[category][provider][id]` nesting
`build_suite` produces) into a `Vector{ResultRow}`. `categories`/`sizes` must match what was
passed to the `build_suite` call that produced `results`, since specs (and therefore
flop/byte counts) are regenerated from `REGISTRY` rather than stored in the result itself.
"""
function resultstable(
        results::BenchmarkGroup; categories = collect(keys(REGISTRY)), sizes = nothing
    )
    rows = ResultRow[]
    for category in categories
        haskey(results, String(category)) || continue
        cases = REGISTRY[category](_sizes_for(sizes, category))
        casesbyid = Dict(c.id => c for c in cases)
        catgroup = results[String(category)]
        for providerlabel in keys(catgroup)
            provgroup = catgroup[providerlabel]
            for id in keys(provgroup)
                trial = provgroup[id]
                case = casesbyid[id]
                mintime = minimum(trial.times)
                memory = trial.memory
                allocs = trial.allocs
                fl = flops(case.spec)
                by = bytes(case.spec)
                gflops = mintime > 0 ? fl / mintime : missing
                gbps = mintime > 0 ? by / mintime : missing
                push!(
                    rows,
                    ResultRow(
                        String(category), providerlabel, id, case.params,
                        mintime, allocs, memory, gflops, gbps
                    )
                )
            end
        end
    end
    return rows
end
