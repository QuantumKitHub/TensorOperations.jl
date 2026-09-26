# Input/output tensors are built in setup=, not timed. Output `C` is preallocated via
# tensoralloc_add/tensoralloc_contract and execution uses the mutating tensor*!, so the timed
# region is the compute kernel, not an allocation -- except NetworkSpec/ncon, which has no
# in-place variant, so its timing includes allocation. BatchedContractSpec's timed region is
# `batch` separate tensorcontract! calls, intentionally (that per-call overhead is the point).

_scalartype_or(::Nothing, provider) = TensorOperations.scalartype(provider)
_scalartype_or(T::Type, provider) = T

function maketensors(spec::AddSpec, provider)
    TA = _scalartype_or(spec.TA, provider)
    TC = _scalartype_or(spec.TC, provider)
    dims = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    A = randtensor(provider, spec.IA, dims, TA)
    pA = TensorOperations.add_indices(spec.IA, spec.IC)
    C = TensorOperations.tensoralloc_add(TC, A, pA, spec.conjA, Val(false), allocator(provider))
    return (C, A, pA)
end

function maketensors(spec::TraceSpec, provider)
    TA = _scalartype_or(spec.TA, provider)
    TC = _scalartype_or(spec.TC, provider)
    dims = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    A = randtensor(provider, spec.IA, dims, TA)
    p, q = TensorOperations.trace_indices(spec.IA, spec.IC)
    C = TensorOperations.tensoralloc_add(TC, A, p, spec.conjA, Val(false), allocator(provider))
    return (C, A, p, q)
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
    return (C, A, pA, B, pB, pAB)
end

function maketensors(spec::BatchedContractSpec, provider)
    TA = _scalartype_or(spec.TA, provider)
    TB = _scalartype_or(spec.TB, provider)
    TC = _scalartype_or(spec.TC, provider)
    dimsA = ntuple(i -> spec.dims[spec.IA[i]], length(spec.IA))
    dimsB = ntuple(i -> spec.dims[spec.IB[i]], length(spec.IB))
    pA, pB, pAB = TensorOperations.contract_indices(spec.IA, spec.IB, spec.IC)
    As = [randtensor(provider, spec.IA, dimsA, TA) for _ in 1:spec.batch]
    Bs = [randtensor(provider, spec.IB, dimsB, TB) for _ in 1:spec.batch]
    Cs = [
        TensorOperations.tensoralloc_contract(
            TC, As[b], pA, spec.conjA, Bs[b], pB, spec.conjB, pAB, Val(false), allocator(provider)
        ) for b in 1:spec.batch
    ]
    return (Cs, As, pA, Bs, pB, pAB)
end

function maketensors(spec::NetworkSpec, provider)
    Ts = something(spec.Ts, fill(TensorOperations.scalartype(provider), length(spec.indexlists)))
    return map(spec.indexlists, Ts) do il, T
        dims = ntuple(i -> spec.dims[abs(il[i])], length(il))
        randtensor(provider, il, dims, T)
    end
end

function execute(spec::AddSpec, (C, A, pA), provider)
    return tensorcopy!(
        C, A, pA, spec.conjA, one(eltype(C)), backend(provider), allocator(provider)
    )
end

function execute(spec::TraceSpec, (C, A, p, q), provider)
    return tensortrace!(
        C, A, p, q, spec.conjA, one(eltype(C)), zero(eltype(C)),
        backend(provider), allocator(provider)
    )
end

function execute(spec::ContractSpec, (C, A, pA, B, pB, pAB), provider)
    return tensorcontract!(
        C, A, pA, spec.conjA, B, pB, spec.conjB, pAB, one(eltype(C)), zero(eltype(C)),
        backend(provider), allocator(provider)
    )
end

function execute(spec::BatchedContractSpec, (Cs, As, pA, Bs, pB, pAB), provider)
    for b in eachindex(Cs)
        tensorcontract!(
            Cs[b], As[b], pA, spec.conjA, Bs[b], pB, spec.conjB, pAB,
            one(eltype(Cs[b])), zero(eltype(Cs[b])), backend(provider), allocator(provider)
        )
    end
    return Cs
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
