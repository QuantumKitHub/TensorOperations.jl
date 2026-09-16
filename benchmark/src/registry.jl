# A category is a generator function `sizes -> Vector{BenchmarkCase}`, registered by name.

"""
    BenchmarkCase(category, id, params, spec)

One concrete benchmark case: `category` groups it (e.g. `:contract`), `id` is a short unique
label within the category, `params` records the sweep parameters that produced it (e.g.
`(; D=64)`), and `spec` is the [`AbstractCaseSpec`](@ref) to execute. `tags` (for
`BenchmarkTools.@tagged`-based filtering, see suite.jl) is computed once here and stored,
rather than re-derived on demand: the category name, plus every `Symbol`-valued entry of
`params` (e.g. `source`, `kind`, `variant`, `layout`, `topic`) -- a generator opts a `params`
field into tag-filtering just by giving it a `Symbol` value.
"""
struct BenchmarkCase
    category::Symbol
    id::String
    params::NamedTuple
    tags::Vector{Any}
    spec::AbstractCaseSpec
end
function BenchmarkCase(category::Symbol, id::AbstractString, params::NamedTuple, spec::AbstractCaseSpec)
    tags = Any[String(category); [string(v) for v in values(params) if v isa Symbol]]
    return BenchmarkCase(category, String(id), params, tags, spec)
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
    within_memory_budget(spec::AbstractCaseSpec, [id]; maxbytes=MAX_CASE_BYTES)

Whether executing `spec` would stay within `maxbytes` of total tensor memory, assuming (in the
absence of an explicit `TA`/`TB`/`TC`/`Ts` override) worst-case `Float64`-sized elements.
Category generators use this to skip dimension/shape combinations that would otherwise OOM the
benchmark process, rather than hand-tuning per-shape size ceilings. Passing `id` additionally
`@warn`s (naming `id`) when a case is skipped, so an unexpectedly sparse sweep is diagnosable
rather than silent.

`MAX_CASE_BYTES` scales with the host's total memory (`Sys.total_memory() ÷ 64`) rather than a
fixed constant, since this ranges from CI runners to large shared workstations.
"""
const MAX_CASE_BYTES = Sys.total_memory() ÷ 64
within_memory_budget(spec::AbstractCaseSpec; maxbytes = MAX_CASE_BYTES) = bytes(spec) <= maxbytes
function within_memory_budget(spec::AbstractCaseSpec, id; maxbytes = MAX_CASE_BYTES)
    ok = within_memory_budget(spec; maxbytes)
    ok || @warn "Skipping benchmark case: exceeds memory budget" id maxbytes bytes = bytes(spec)
    return ok
end
