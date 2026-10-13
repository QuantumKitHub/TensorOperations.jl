# Threading is process-wide, not a provider/spec concern, and deliberately not baked into
# per-case timing (that would count the thread-count switch as part of the measurement).
# `Strided.set_num_threads` is capped by `Threads.nthreads()` (fixed at Julia startup); sweeping
# above it needs relaunching Julia -- see scripts/run_benchmarks.jl.

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

Set `BLAS`/`Strided` thread counts according to `cfg`, permanently. For one-shot,
process-level configuration (`benchmarks.jl` calls this before building `SUITE`).
"""
function set_threads!(cfg::ThreadConfig)
    cfg.blas === nothing || BLAS.set_num_threads(cfg.blas)
    cfg.strided === nothing || Strided.set_num_threads(cfg.strided)
    return nothing
end

"""
    with_threads(f, cfg::ThreadConfig)

Run `f()` under `cfg`, restoring prior thread counts afterwards. For comparing a couple of
configs around a whole `run(suite)` call -- never around a single `@benchmarkable` case.
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
