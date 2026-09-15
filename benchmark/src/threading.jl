# Threading is orthogonal to *which* tensor type/backend a provider uses (it applies
# process-wide via BLAS and Strided's own runtime thread counters), so it is not part of the
# `AbstractProvider` interface, and -- importantly -- it is NOT baked into individual
# `@benchmarkable` cases either: doing that would mean every timed sample pays for
# `BLAS.set_num_threads`/`Strided.set_num_threads` plus a closure allocation, polluting the very
# measurement it's supposed to control. Instead, a thread configuration is applied *once*,
# around an entire suite (or PkgBenchmark) run -- see `set_threads!`/`with_threads` below, and
# `benchmarks.jl`, which calls `set_threads!` once at the top level before building `SUITE`.
#
# Note: `Strided.set_num_threads` is capped by `Threads.nthreads()`, which is fixed at Julia
# process startup (`-t`/`JULIA_NUM_THREADS`). Sweeping *above* the process's thread count is not
# possible at runtime; `scripts/run_benchmarks.jl` covers that case by relaunching Julia with a
# different `-t` per outer thread count (via `PkgBenchmark`'s `juliacmd`).

"""
    ThreadConfig(; blas=nothing, strided=nothing)

A named thread-count configuration. `nothing` for either field means "leave that thread count
untouched". Use [`with_threads`](@ref) to apply one around a block of code.
"""
struct ThreadConfig
    blas::Union{Nothing, Int}
    strided::Union{Nothing, Int}
    name::String
end
function ThreadConfig(; blas::Union{Nothing, Int} = nothing, strided::Union{Nothing, Int} = nothing, name = nothing)
    autoname = "blas=$(something(blas, "-")),strided=$(something(strided, "-"))"
    return ThreadConfig(blas, strided, something(name, autoname))
end

label(cfg::ThreadConfig) = cfg.name

"""
    set_threads!(cfg::ThreadConfig)

Set `BLAS`/`Strided` thread counts according to `cfg`, permanently (i.e. not restored
afterwards). Intended for one-shot, process-level configuration -- e.g. `benchmarks.jl` calls
this once, before `SUITE` is built, so that the thread-count choice is entirely outside of
anything ever timed.
"""
function set_threads!(cfg::ThreadConfig)
    cfg.blas === nothing || BLAS.set_num_threads(cfg.blas)
    cfg.strided === nothing || Strided.set_num_threads(cfg.strided)
    return nothing
end

"""
    with_threads(f, cfg::ThreadConfig)

Run `f()` with `BLAS`/`Strided` thread counts set according to `cfg`, restoring the prior
counts afterwards (even if `f` throws). Intended for interactively comparing a couple of
thread counts around a whole `run(suite)`/`benchmarkpkg(...)` call -- never around a single
`@benchmarkable` case, since that would count the thread-count switch itself as part of the
timed operation.
"""
function with_threads(f, cfg::ThreadConfig)
    oldblas = BLAS.get_num_threads()
    oldstrided = Strided.get_num_threads()
    try
        set_threads!(cfg)
        return f()
    finally
        BLAS.set_num_threads(oldblas)
        Strided.set_num_threads(oldstrided)
    end
end
