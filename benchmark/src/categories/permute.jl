# Permutation-only cost (`tensorcopy!`), isolated from contraction cost. Rank-4 tensor, sweeping
# leg dimension x how scrambled the permutation is.

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

register_category!(:permute, _permute_cases; sizes = (4, 6, 8, 15, 32, 63, 96, 128, 200, 256))
