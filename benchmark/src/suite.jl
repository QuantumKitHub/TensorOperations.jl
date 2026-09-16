# Nests a BenchmarkGroup as [category][provider label][case id]. No threading axis -- see threading.jl.

"""
    build_suite(providers; categories=collect(keys(REGISTRY)), sizes=nothing, casefilter=nothing)

Build a `BenchmarkTools.BenchmarkGroup` covering every registered category (or the subset in
`categories`) for every provider in `providers` (skipping providers that opt out via
[`supports`](@ref)).

`sizes` may be `nothing` (use each category's [`default_sizes`](@ref)), a size sweep applied to
every category, or a `Dict{Symbol}` mapping category name to its own size sweep.

`casefilter` (a `BenchmarkCase -> Bool` predicate, or `nothing`) selects a subset of cases
within each category -- e.g. `casefilter = c -> c.params.source == :tccg` runs only the
TCCG-sourced cases within `:pairwise`, without needing a separate category.
"""
function build_suite(
        providers::AbstractVector{<:AbstractProvider};
        categories = collect(keys(REGISTRY)),
        sizes = nothing,
        casefilter = nothing
    )
    suite = BenchmarkGroup()
    for category in categories
        generator = REGISTRY[category]
        catsizes = _sizes_for(sizes, category)
        cases = generator(catsizes)
        casefilter === nothing || (cases = filter(casefilter, cases))
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
