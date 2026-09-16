# Pairwise contractions: a synthetic parametric shape family, plus the real TCCG equations
# (tccg.jl) -- both just `ContractSpec`, merged into one category, tagged (`:synthetic`/`:tccg`)
# for filtering via `@tagged` (see registry.jl). Sizes mix power-of-two with off-by-one values.
#
# Each shape gets up to 4 label-order "layouts": `(openA...,contract...)`/`(contract...,openB...)`
# is directly reshapeable to GEMM (no data movement); interleaving open and contracted labels
# (e.g. `[a1,c1,a2,c2]`) cannot be expressed as a single reshape or BLAS transpose flag, forcing
# a real permutation -- exactly the case that separates StridedNative from StridedBLAS. Layouts
# that coincide with the GEMM one (e.g. when a shape has ≤1 contracted index) are skipped.

const CONTRACT_SHAPES = (
    (1, 1, 1),   # matrix-vector-like
    (2, 1, 2),   # single shared bond, several open legs each side
    (2, 2, 2),   # GEMM-like, rank 4 total
    (1, 3, 1),   # trace-heavy: many contracted, few open
    (1, 0, 1),   # pure outer product, no contraction
)

function _interleave(a::Vector{Symbol}, b::Vector{Symbol})
    n = min(length(a), length(b))
    return vcat((Symbol[a[i], b[i]] for i in 1:n)..., a[(n + 1):end], b[(n + 1):end])
end

function _contract_layouts(openA, contract, openB)
    gemmA, gemmB = vcat(openA, contract), vcat(contract, openB)
    permA, permB = _interleave(openA, contract), _interleave(contract, openB)
    candidates = (
        (:gemm_ready, gemmA, gemmB),
        (:a_permuted, permA, gemmB),
        (:b_permuted, gemmA, permB),
        (:both_permuted, permA, permB),
    )
    seen = Set{Tuple{Vector{Symbol}, Vector{Symbol}}}()
    layouts = Tuple{Symbol, Vector{Symbol}, Vector{Symbol}}[]
    for (layout, IA, IB) in candidates
        (IA, IB) in seen && continue
        push!(seen, (IA, IB))
        push!(layouts, (layout, IA, IB))
    end
    return layouts
end

function _synthetic_contract_cases(sizes)
    cases = BenchmarkCase[]
    for dim in sizes
        for (nopenA, ncontract, nopenB) in CONTRACT_SHAPES
            openA = [Symbol("a", i) for i in 1:nopenA]
            contract = [Symbol("c", i) for i in 1:ncontract]
            openB = [Symbol("b", i) for i in 1:nopenB]
            IC = vcat(openA, openB)
            for (layout, IA, IB) in _contract_layouts(openA, contract, openB)
                dims = Dict{Symbol, Int}(l => dim for l in vcat(IA, IB))
                spec = ContractSpec(IA, IB, IC, dims)
                id = "dim$(dim)_$(nopenA)_$(ncontract)_$(nopenB)_$(layout)"
                within_memory_budget(spec, id) || continue
                params = (; dim, nopenA, ncontract, nopenB, layout, source = :synthetic)
                push!(cases, BenchmarkCase(:contract, id, params, spec))
            end
        end
    end
    return cases
end

_contract_cases(sizes) = vcat(_synthetic_contract_cases(sizes), _tccg_cases(sizes))

register_category!(
    :contract, _contract_cases;
    sizes = (8, 12, 15, 16, 24, 32, 63, 96, 128, 200, 256)
)
