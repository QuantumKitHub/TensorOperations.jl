# Trace-heavy benchmarks: partial traces (some legs traced, some kept open) and full traces
# (reduce all the way to a scalar), as distinct from the pairwise-contraction category.
#
# `sizes` is a list of leg dimensions; for each dimension we benchmark a rank-6 tensor traced
# down to rank-2 (partial trace) and a rank-4 tensor traced all the way to a scalar (full
# trace).

function _trace_cases(sizes)
    cases = BenchmarkCase[]
    for dim in sizes
        # partial trace: rank 6 -> rank 2, trace 2 pairs, keep 2 open
        IA6 = [:o1, :o2, :t1, :t1, :t2, :t2]
        IC6 = [:o1, :o2]
        dims6 = Dict{Symbol, Int}(l => dim for l in unique(IA6))
        partialspec = TraceSpec(IA6, IC6, dims6)
        if within_memory_budget(partialspec)
            push!(
                cases,
                BenchmarkCase(:trace, "partial_dim$(dim)", (; dim, kind = :partial), partialspec)
            )
        end

        # full trace: rank 4 -> scalar
        IA4 = [:t1, :t1, :t2, :t2]
        IC4 = Symbol[]
        dims4 = Dict{Symbol, Int}(l => dim for l in unique(IA4))
        fullspec = TraceSpec(IA4, IC4, dims4)
        if within_memory_budget(fullspec)
            push!(cases, BenchmarkCase(:trace, "full_dim$(dim)", (; dim, kind = :full), fullspec))
        end
    end
    return cases
end

register_category!(:trace, _trace_cases; sizes = (8, 15, 32, 63, 96, 128, 200, 256))
