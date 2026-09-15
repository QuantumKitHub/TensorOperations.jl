# Real quantum-chemistry contractions from the TCCG benchmark (github.com/HPAC/tccg): CCSD,
# CCSD(T), AO2MO, INTENSLI, as "C-A-B" index strings (e.g. "ij-ik-kj" = C[i,j]=A[i,k]*B[k,j]).
# `sizes` applies one leg dimension uniformly to every index letter, as TCCG itself does.

const TCCG_CONTRACTIONS = (
    # CCSD
    (id = "ccsd_1", C = "ij", A = "ik", B = "kj"),
    (id = "ccsd_2", C = "ij", A = "ikl", B = "ljk"),
    (id = "ccsd_3", C = "ij", A = "kil", B = "lkj"),
    (id = "ccsd_4", C = "ijk", A = "ikl", B = "lj"),
    (id = "ccsd_5", C = "ijk", A = "ilk", B = "jl"),
    (id = "ccsd_6", C = "ijk", A = "ilmk", B = "mjl"),
    (id = "ccsd_7", C = "ijkl", A = "imjn", B = "lnkm"),
    (id = "ccsd_8", C = "ijkl", A = "imjn", B = "nlmk"),
    (id = "ccsd_9", C = "ijkl", A = "minl", B = "njmk"),
    # CCSD(T)
    (id = "ccsd_t_1", C = "abcijk", A = "ijma", B = "mkbc"),
    (id = "ccsd_t_2", C = "abcijk", A = "ijmb", B = "mkac"),
    (id = "ccsd_t_3", C = "abcijk", A = "ijmc", B = "mkab"),
    (id = "ccsd_t_4", C = "abcijk", A = "ikmb", B = "mjac"),
    # AO2MO integral transformation
    (id = "ao2mo_1", C = "aqrs", A = "pa", B = "pqrs"),
    (id = "ao2mo_2", C = "abrs", A = "qb", B = "aqrs"),
    (id = "ao2mo_3", C = "abcs", A = "rc", B = "abrs"),
    # INTENSLI
    (id = "intensli_1", C = "abj", A = "bka", B = "kj"),
    (id = "intensli_2", C = "ajb", A = "kba", B = "jk"),
    (id = "intensli_3", C = "abjc", A = "cbka", B = "kj"),
    (id = "intensli_4", C = "ajbc", A = "ckba", B = "jk"),
    (id = "intensli_5", C = "abjc", A = "kbac", B = "jk"),
    (id = "intensli_6", C = "abjcd", A = "dkbac", B = "kj"),
    (id = "intensli_7", C = "adbjc", A = "cbdka", B = "kj"),
    (id = "intensli_8", C = "ajbdc", A = "ckbad", B = "jk"),
)

_tccg_labels(s::AbstractString) = [Symbol(c) for c in s]

function _tccg_cases(sizes)
    cases = BenchmarkCase[]
    for dim in sizes
        for eq in TCCG_CONTRACTIONS
            IA, IB, IC = _tccg_labels(eq.A), _tccg_labels(eq.B), _tccg_labels(eq.C)
            dims = Dict{Symbol, Int}(l => dim for l in vcat(IA, IB))
            spec = ContractSpec(IA, IB, IC, dims)
            within_memory_budget(spec) || continue
            id = "$(eq.id)_dim$(dim)"
            push!(cases, BenchmarkCase(:tccg, id, (; dim, equation = eq.id), spec))
        end
    end
    return cases
end

register_category!(:tccg, _tccg_cases; sizes = (4, 8, 12, 16, 24, 32))
