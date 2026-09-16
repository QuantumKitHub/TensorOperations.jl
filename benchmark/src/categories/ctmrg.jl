# CTMRG corner-growth step (boundary-MPS method for 2D PEPS): C-T-T-a, producing an unfused
# rank-4 corner (chi,chi,D2,D2). Cost ~ O(chi^3*D2^3) (literature O(chi^3*D^6), D2=D^2).
# `sizes` sweeps environment bond `chi`; PEPS bond `D` is fixed. Merged into `:network` (see
# network.jl) tagged `params.topic = :ctmrg`.

const CTMRG_PEPS_BOND = 3   # D

function _ctmrg_case(chi)
    D2 = CTMRG_PEPS_BOND^2
    indexlists = [
        [1, 2],            # C: (chi, chi)
        [1, 3, -10],       # T_left: (chi, D2, chi)
        [2, 4, -11],       # T_top: (chi, D2, chi)
        [3, -12, 4, -13],  # a (double-layer PEPS tensor): (D2, D2, D2, D2)
    ]
    dims = Dict(1 => chi, 2 => chi, 3 => D2, 4 => D2, 10 => chi, 11 => chi, 12 => D2, 13 => D2)
    spec = NetworkSpec(indexlists, dims; output = [-10, -11, -12, -13])
    params = (; chi, D = CTMRG_PEPS_BOND, topic = :ctmrg)
    return BenchmarkCase(:network, "ctmrg_corner_chi$(chi)", params, spec)
end

_ctmrg_cases(sizes) = [_ctmrg_case(chi) for chi in sizes]
