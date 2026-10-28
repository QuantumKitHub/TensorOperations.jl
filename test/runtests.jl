using ParallelTestRunner
using TensorOperations

testsuite = find_tests(@__DIR__)

# --fast is passed by the CI on draft PRs, but there are no slow tests to skip
args = parse_args(ARGS; custom = ["fast"])
filter_tests!(testsuite, args)

# don't run all tests on GPU, only the GPU specific ones
if get(ENV, "BUILDKITE", "false") == "true"
    filter!(in(("extensions/cutensor", "extensions/gpuarrays")) ∘ first, testsuite)
end
Sys.isapple() && get(ENV, "CI", "false") == "true" && delete!(testsuite, "ad/enzyme")
# the Windows build of TBLIS can only be tested for refusing to run
delete!(testsuite, Sys.iswindows() ? "extensions/tblis" : "extensions/tblis_unsupported")

# The cuTENSOR and Bumper tests first check that the macros requiring them throw when they are
# not loaded, so they run on a fresh worker rather than one that may have loaded them already
const fresh_worker_tests = ("extensions/cutensor", "extensions/bumper")
const fresh_workers = Channel{ParallelTestRunner.PTRWorker}(Inf)
function test_worker(name)
    name in fresh_worker_tests || return nothing
    worker = addworker()
    put!(fresh_workers, worker)
    return worker
end

const init_code = quote
    using TensorOperations
    using LinearAlgebra
    using Test
    using Random
    Random.seed!(1234567)

    const is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

    using TensorOperations: IndexError
    using TensorOperations: BaseCopy, BaseView, StridedNative, StridedBLAS
    using TensorOperations: DefaultAllocator, ManualAllocator, BufferAllocator

    precision(::Type{<:Union{Float32, Complex{Float32}}}) = 1.0e-2
    precision(::Type{<:Union{Float64, Complex{Float64}}}) = 1.0e-8

    # https://github.com/QuantumKitHub/TensorOperations.jl/issues/280: the generated code has to
    # keep the user's `LineNumberNode`s -- Julia only emits a coverage counter for lines that a
    # `LineNumberNode` points at -- and drop the ones synthesized by the parser, which would
    # otherwise re-attribute the statements that follow them to a line in TensorOperations itself.
    # Only `LineNumberNode`s in block-statement position matter for that: elsewhere (most notably
    # the mandatory 2nd argument of a `:macrocall`) they are structurally required and left alone.
    function statementlinenumbernodes(ex, acc = LineNumberNode[])
        if ex isa Expr
            if ex.head === :block
                for e in ex.args
                    e isa LineNumberNode && push!(acc, e)
                end
            end
            foreach(e -> statementlinenumbernodes(e, acc), ex.args)
        end
        return acc
    end
end

try
    runtests(TensorOperations, args; testsuite, init_code, test_worker)
finally
    close(fresh_workers)
    foreach(ParallelTestRunner.Malt.stop, fresh_workers)
end
