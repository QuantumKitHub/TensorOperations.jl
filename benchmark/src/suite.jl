# Suite assembly: nests a `BenchmarkGroup` as [category][provider label][case id]. Threading
# is deliberately NOT an axis here -- see threading.jl -- since applying it per-case would
# count the thread-count switch as part of the timed operation. `benchmarks.jl` (the
# PkgBenchmark entrypoint) is expected to be a thin wrapper: call `set_threads!` once, construct
# the `AbstractProvider`s to compare, call `build_suite`, assign the result to `const SUITE`.

"""
    build_suite(providers; categories=collect(keys(REGISTRY)), sizes=nothing)

Build a `BenchmarkTools.BenchmarkGroup` covering every registered category (or the subset in
`categories`) for every provider in `providers` (skipping providers that opt out via
[`supports`](@ref)).

`sizes` may be `nothing` (use each category's [`default_sizes`](@ref)), a size sweep applied to
every category, or a `Dict{Symbol}` mapping category name to its own size sweep.
"""
function build_suite(
        providers::AbstractVector{<:AbstractProvider};
        categories = collect(keys(REGISTRY)),
        sizes = nothing
    )
    suite = BenchmarkGroup()
    for category in categories
        generator = REGISTRY[category]
        catsizes = _sizes_for(sizes, category)
        cases = generator(catsizes)
        catgroup = suite[String(category)] = BenchmarkGroup()
        for provider in providers
            supports(provider, category) || continue
            provgroup = catgroup[label(provider)] = BenchmarkGroup()
            for case in cases
                provgroup[case.id] = make_benchmarkable(case, provider)
            end
        end
    end
    return suite
end

_sizes_for(::Nothing, category::Symbol) = default_sizes(category)
_sizes_for(sizes::AbstractDict, category::Symbol) = get(sizes, category, default_sizes(category))
_sizes_for(sizes, ::Symbol) = sizes
