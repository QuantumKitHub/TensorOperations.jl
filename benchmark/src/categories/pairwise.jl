# Pairwise contractions: a synthetic parametric shape family, plus the real TCCG equations
# (tccg.jl) -- both just `ContractSpec`, merged into one category and distinguished via
# `params.source` (`:synthetic`/`:tccg`) so `build_suite`'s `casefilter` can select either.
# Sizes mix power-of-two with off-by-one/arbitrary values to catch alignment effects.

const PAIRWISE_SHAPES = (
    (1, 1, 1),   # matrix-vector-like
    (2, 1, 2),   # single shared bond, several open legs each side
    (2, 2, 2),   # GEMM-like, rank 4 total
    (1, 3, 1),   # trace-heavy: many contracted, few open
    (1, 0, 1),   # pure outer product, no contraction
)

function _synthetic_pairwise_cases(sizes)
    cases = BenchmarkCase[]
    for dim in sizes
        for (nopenA, ncontract, nopenB) in PAIRWISE_SHAPES
            openA = [Symbol("a", i) for i in 1:nopenA]
            contract = [Symbol("c", i) for i in 1:ncontract]
            openB = [Symbol("b", i) for i in 1:nopenB]
            IA = vcat(openA, contract)
            IB = vcat(contract, openB)
            IC = vcat(openA, openB)
            dims = Dict{Symbol, Int}(l => dim for l in vcat(IA, IB))
            spec = ContractSpec(IA, IB, IC, dims)
            within_memory_budget(spec) || continue
            id = "dim$(dim)_$(nopenA)_$(ncontract)_$(nopenB)"
            params = (; dim, nopenA, ncontract, nopenB, source = :synthetic)
            push!(cases, BenchmarkCase(:pairwise, id, params, spec))
        end
    end
    return cases
end

_pairwise_cases(sizes) = vcat(_synthetic_pairwise_cases(sizes), _tccg_cases(sizes))

register_category!(
    :pairwise, _pairwise_cases;
    sizes = (8, 12, 15, 16, 24, 32, 63, 96, 128, 200, 256)
)
