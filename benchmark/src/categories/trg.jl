# TRG plaquette contraction: a 4-ring of rank-3 tensors (from SVD-splitting neighboring rank-4
# TRG tensors), producing the coarse-grained rank-4 tensor. Cost ~ O(chi^6). `sizes` sweeps the
# bond dimension `chi`. Merged into `:network` (see network.jl) tagged `params.topic = :trg`.

function _trg_case(chi)
    indexlists = [
        [-10, 1, 2],
        [2, -11, 3],
        [3, -12, 4],
        [4, -13, 1],
    ]
    dims = Dict(1 => chi, 2 => chi, 3 => chi, 4 => chi, 10 => chi, 11 => chi, 12 => chi, 13 => chi)
    spec = NetworkSpec(indexlists, dims; output = [-10, -11, -12, -13])
    params = (; chi, topic = :trg)
    return BenchmarkCase(:network, "trg_plaquette_chi$(chi)", params, spec)
end

_trg_cases(sizes) = [_trg_case(chi) for chi in sizes]
