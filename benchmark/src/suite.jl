# Nests a BenchmarkGroup as [category][provider label][case id]["benchmark"]. Each case-id leaf
# is itself a tagged BenchmarkGroup (tags from `casetags`: category + Symbol-valued params), so
# `suite[BenchmarkTools.@tagged "tccg"]` (or any boolean tag expression) selects a subset of
# cases natively -- no bespoke filter predicate needed. No threading axis -- see threading.jl.

"""
    build_suite(providers; categories=collect(keys(REGISTRY)), sizes=nothing)

Build a `BenchmarkTools.BenchmarkGroup` covering every registered category (or the subset in
`categories`) for every provider in `providers` (skipping providers that opt out via
[`supports`](@ref)).

`sizes` may be `nothing` (use each category's [`default_sizes`](@ref)), a size sweep applied to
every category, or a `Dict{Symbol}` mapping category name to its own size sweep.

To run only a tagged subset, filter *after* building: `suite[@tagged "tccg"]` (re-exported from
BenchmarkTools), or combine tags: `suite[@tagged "pairwise" && "tccg"]`.
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
                provgroup[case.id] = BenchmarkGroup(
                    casetags(case), "benchmark" => make_benchmarkable(case, provider)
                )
            end
        end
    end
    return suite
end

_sizes_for(::Nothing, category::Symbol) = default_sizes(category)
_sizes_for(sizes::AbstractDict, category::Symbol) = get(sizes, category, default_sizes(category))
_sizes_for(sizes, ::Symbol) = sizes
