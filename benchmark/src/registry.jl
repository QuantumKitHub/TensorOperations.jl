# A category is registered as a single generator function `sizes -> Vector{BenchmarkCase}`.
# Adding a new category to the suite is exactly: write one file defining a generator, `include`
# it, and call `register_category!` -- nothing else in the framework changes.

"""
    BenchmarkCase(category, id, params, spec)

One concrete benchmark case: `category` groups it (e.g. `:pairwise`), `id` is a short unique
label within the category, `params` records the sweep parameters that produced it (e.g.
`(; D=64)`), and `spec` is the [`AbstractCaseSpec`](@ref) to execute.
"""
struct BenchmarkCase
    category::Symbol
    id::String
    params::NamedTuple
    spec::AbstractCaseSpec
end

const REGISTRY = Dict{Symbol, Function}()
const DEFAULT_SIZES = Dict{Symbol, Any}()

"""
    register_category!(name::Symbol, generator::Function; sizes=(4, 8, 16, 32, 64, 128))

Register `generator(sizes) -> Vector{BenchmarkCase}` under category `name`, along with the
default size sweep to use when the caller doesn't supply one. Calling this again for the same
`name` overwrites the previous generator/default.
"""
function register_category!(name::Symbol, generator::Function; sizes = (4, 8, 16, 32, 64, 128))
    REGISTRY[name] = generator
    DEFAULT_SIZES[name] = sizes
    return nothing
end

"""
    default_sizes(category::Symbol)

The default size sweep passed to `category`'s generator when the caller doesn't supply one.
Each category interprets `sizes` in its own way (a list of ranks, of bond dimensions, ...).
"""
default_sizes(category::Symbol) = get(DEFAULT_SIZES, category, (4, 8, 16, 32, 64, 128))

"""
    within_memory_budget(spec::AbstractCaseSpec; maxbytes=MAX_CASE_BYTES)

Whether executing `spec` would stay within `maxbytes` of total tensor memory, assuming (in the
absence of an explicit `TA`/`TB`/`TC`/`Ts` override) worst-case `Float64`-sized elements.
Category generators use this to silently skip dimension/shape combinations that would
otherwise OOM the benchmark process, rather than hand-tuning per-shape size ceilings.
"""
const MAX_CASE_BYTES = 2^28  # 256 MiB
within_memory_budget(spec::AbstractCaseSpec; maxbytes = MAX_CASE_BYTES) = bytes(spec) <= maxbytes
