# Contractions with differing input/output element types (promote_add/promote_contract).

const MIXED_PRECISION_COMBOS = (
    (Float32, Float32, Float64),     # low-precision inputs, high-precision accumulation
    (Float64, ComplexF64, ComplexF64),  # mixed real/complex
    (Float64, ComplexF64, ComplexF32),  # mixed real/complex, downcast output
)

function _mixed_precision_cases(sizes)
    cases = BenchmarkCase[]
    for dim in sizes
        for (TA, TB, TC) in MIXED_PRECISION_COMBOS
            IA = [:a1, :c1]
            IB = [:c1, :b1]
            IC = [:a1, :b1]
            dims = Dict{Symbol, Int}(:a1 => dim, :b1 => dim, :c1 => dim)
            spec = ContractSpec(IA, IB, IC, dims; TA, TB, TC)
            within_memory_budget(spec) || continue
            id = "dim$(dim)_$(TA)_$(TB)_$(TC)"
            push!(cases, BenchmarkCase(:mixed_precision, id, (; dim, TA, TB, TC), spec))
        end
    end
    return cases
end

register_category!(:mixed_precision, _mixed_precision_cases; sizes = (33, 129, 500))
