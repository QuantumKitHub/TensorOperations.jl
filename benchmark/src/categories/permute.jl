# Permutation-only benchmarks (transpose cost), a first-class category in its own right:
# TBLIS/TCL-style backends exist precisely because transpose-free vs. transpose-then-GEMM
# strategies differ, so isolating pure `tensorcopy!` cost from contraction cost matters.
#
# `sizes` is a list of leg dimensions; for each dimension we benchmark permutations of a
# rank-4 tensor across a spread of "how scrambled" the permutation is (identity-adjacent vs.
# fully reversed).

const PERMUTE_PATTERNS = (
    [1, 2, 3, 4],   # identity (still exercises the copy machinery, no real permutation)
    [2, 1, 3, 4],   # single adjacent swap
    [4, 3, 2, 1],   # full reversal
    [3, 1, 4, 2],   # scrambled
)

function _permute_cases(sizes)
    cases = BenchmarkCase[]
    for dim in sizes
        IA = [Symbol("a", i) for i in 1:4]
        dims = Dict{Symbol, Int}(l => dim for l in IA)
        for pattern in PERMUTE_PATTERNS
            IC = IA[pattern]
            spec = AddSpec(IA, IC, dims)
            within_memory_budget(spec) || continue
            id = "dim$(dim)_perm$(join(pattern))"
            push!(cases, BenchmarkCase(:permute, id, (; dim, pattern), spec))
        end
    end
    return cases
end

register_category!(:permute, _permute_cases; sizes = (8, 32, 64, 128, 256))
