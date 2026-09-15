# Turns a (spec, provider) pair into an executable BenchmarkTools benchmark. Everything that
# isn't the operation itself -- input tensors, the *output* tensor, and the low-level
# `Index2Tuple` index-permutation computation -- is built fresh in the `setup=` block (once per
# sample, not per eval), so it doesn't count against the timed operation. In particular, `C` is
# preallocated (via `tensoralloc_add`/`tensoralloc_contract`, so it goes through the provider's
# configured allocator) and execution uses the mutating `tensorcopy!`/`tensortrace!`/
# `tensorcontract!`, so the timed region is exactly the compute kernel, not an output
# allocation.
#
# `NetworkSpec` is the one exception: `ncon` has no public in-place variant (it always
# allocates its final and intermediate results internally), so its timed region does include
# allocation. Reimplementing `ncon`'s tree contraction manually with preallocated buffers would
# let us avoid that, but is out of scope for v1 -- `ncon`-based network cases should be read as
# "cost of ncon", allocation included, not "cost of the raw contraction kernel".

_scalartype_or(::Nothing, provider) = TensorOperations.scalartype(provider)
_scalartype_or(T::Type, provider) = T

function maketensors(spec::AddSpec, provider)
    TA = _scalartype_or(spec.TA, provider)
    TC = _scalartype_or(spec.TC, provider)
    dims = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    A = randtensor(provider, spec.IA, dims, TA)
    pA = TensorOperations.add_indices(spec.IA, spec.IC)
    C = TensorOperations.tensoralloc_add(TC, A, pA, spec.conjA, Val(false), allocator(provider))
    return (A, pA, C)
end

function maketensors(spec::TraceSpec, provider)
    TA = _scalartype_or(spec.TA, provider)
    TC = _scalartype_or(spec.TC, provider)
    dims = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    A = randtensor(provider, spec.IA, dims, TA)
    p, q = TensorOperations.trace_indices(spec.IA, spec.IC)
    C = TensorOperations.tensoralloc_add(TC, A, p, spec.conjA, Val(false), allocator(provider))
    return (A, p, q, C)
end

function maketensors(spec::ContractSpec, provider)
    TA = _scalartype_or(spec.TA, provider)
    TB = _scalartype_or(spec.TB, provider)
    TC = _scalartype_or(spec.TC, provider)
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    A = randtensor(provider, spec.IA, dimsA, TA)
    B = randtensor(provider, spec.IB, dimsB, TB)
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    C = TensorOperations.tensoralloc_contract(
        TC, A, pA, spec.conjA, B, pB, spec.conjB, pAB, Val(false), allocator(provider)
    )
    return (A, B, pA, pB, pAB, C)
end

function maketensors(spec::NetworkSpec, provider)
    Ts = something(spec.Ts, fill(TensorOperations.scalartype(provider), length(spec.indexlists)))
    return map(spec.indexlists, Ts) do il, T
        dims = ntuple(i -> spec.dims[abs(il[i])], length(il))
        randtensor(provider, il, dims, T)
    end
end

function execute(spec::AddSpec, (A, pA, C), provider)
    return tensorcopy!(
        C, A, pA, spec.conjA, one(eltype(C)), backend(provider), allocator(provider)
    )
end

function execute(spec::TraceSpec, (A, p, q, C), provider)
    return tensortrace!(
        C, A, p, q, spec.conjA, one(eltype(C)), zero(eltype(C)),
        backend(provider), allocator(provider)
    )
end

function execute(spec::ContractSpec, (A, B, pA, pB, pAB, C), provider)
    return tensorcontract!(
        C, A, pA, spec.conjA, B, pB, spec.conjB, pAB, one(eltype(C)), zero(eltype(C)),
        backend(provider), allocator(provider)
    )
end

function execute(spec::NetworkSpec, tensors, provider)
    return ncon(
        tensors, spec.indexlists, spec.conjlist;
        order = spec.order, output = spec.output,
        backend = backend(provider), allocator = allocator(provider)
    )
end

"""
    make_benchmarkable(case::BenchmarkCase, provider::AbstractProvider)

Build a `BenchmarkTools.Benchmark` for `case` run against `provider`, constructing fresh
input tensors and a fresh (preallocated, uninitialized) output tensor before every sample.
"""
function make_benchmarkable(case::BenchmarkCase, provider::AbstractProvider)
    spec = case.spec
    return @benchmarkable(
        execute($spec, ts, $provider),
        setup = (ts = maketensors($spec, $provider))
    )
end
