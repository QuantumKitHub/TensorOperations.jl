# Pairwise contractions: a synthetic parametric shape family, the real TCCG equations
# (tccg.jl), and a batch of independent small contractions -- all just `ContractSpec`/
# `BatchedContractSpec`, merged into one category, tagged (`:synthetic`/`:tccg`/`:batched`) for
# filtering via `@tagged` (see registry.jl). Sizes mix power-of-two with off-by-one values.
#
# Each shape gets up to 5 label-order "layouts" (see tenferro-rs's binary_diagnostic suite and
# TAPP's index taxonomy for the shapes this is drawn from):
#   - `:gemm_ready`: `(openA...,contract...)`/`(contract...,openB...)`, reshapeable to GEMM with
#     no data movement.
#   - `:a_permuted`/`:b_permuted`/`:both_permuted`: open and contracted labels interleaved (e.g.
#     `[a1,c1,a2,c2]`), which cannot be expressed as a single reshape or BLAS transpose flag,
#     forcing a real permutation -- exactly what separates StridedNative from StridedBLAS.
#   - `:contract_scrambled`: contracted indices form a contiguous block in *both* operands (so
#     naively "looks" reshapeable) but in a *different relative order* between A and B -- e.g.
#     TAPP's `D[a,d,e] = A[a,b,c]*B[c,d,e,b]`. No single reshape+transpose pair fixes this since
#     the shared multi-index isn't flattened consistently between operands.
# Layouts that coincide with an earlier one (e.g. when a shape has ≤1 contracted index) are
# skipped by the dedup below.

const CONTRACT_SHAPES = (
    (1, 1, 1),   # matrix-vector-like
    (2, 1, 2),   # single shared bond, several open legs each side
    (2, 2, 2),   # GEMM-like, rank 4 total
    (1, 3, 1),   # trace-heavy: many contracted, few open
    (1, 0, 1),   # pure outer product, no contraction
    (0, 2, 3),   # environment-into-tensor: A fully absorbed, no open legs of its own
    (3, 2, 3),   # high total rank (8), multiple bonds -- TBLIS/TAPP-style bundle-of-dims shape
)

function _interleave(a::Vector{Symbol}, b::Vector{Symbol})
    n = min(length(a), length(b))
    return vcat((Symbol[a[i], b[i]] for i in 1:n)..., a[(n + 1):end], b[(n + 1):end])
end

function _contract_layouts(openA, contract, openB)
    gemmA, gemmB = vcat(openA, contract), vcat(contract, openB)
    permA, permB = _interleave(openA, contract), _interleave(contract, openB)
    scrambledB = vcat(reverse(contract), openB)
    candidates = (
        (:gemm_ready, gemmA, gemmB),
        (:a_permuted, permA, gemmB),
        (:b_permuted, gemmA, permB),
        (:both_permuted, permA, permB),
        (:contract_scrambled, gemmA, scrambledB),
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

# Batch of independent small contractions (tenferro-rs's `bij,bjk->bik`-style motif): no fused
# batched-GEMM primitive exists here, so this is `batch` real dispatches, not one bigger call --
# a distinct, call-overhead-dominated regime from the single large `ContractSpec` cases above.
const BATCH_SIZES = (4, 16, 64)

function _batched_contract_cases(sizes)
    cases = BenchmarkCase[]
    IA, IB, IC = [:a1, :c1], [:c1, :b1], [:a1, :b1]
    for dim in sizes
        dims = Dict(:a1 => dim, :b1 => dim, :c1 => dim)
        for batch in BATCH_SIZES
            spec = BatchedContractSpec(batch, IA, IB, IC, dims)
            id = "batched_dim$(dim)_batch$(batch)"
            within_memory_budget(spec, id) || continue
            params = (; dim, batch, source = :batched)
            push!(cases, BenchmarkCase(:contract, id, params, spec))
        end
    end
    return cases
end

_contract_cases(sizes) = vcat(
    _synthetic_contract_cases(sizes), _tccg_cases(sizes), _batched_contract_cases(sizes)
)

register_category!(
    :contract, _contract_cases;
    sizes = (8, 12, 15, 16, 24, 32, 63, 96, 128, 200, 256)
)
