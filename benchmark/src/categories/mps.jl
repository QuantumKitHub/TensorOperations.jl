# The MPS/MPO DMRG effective-Hamiltonian motif: applying `H_eff = L - W - R` to an MPS
# tensor, i.e. `environment(D,D,w) x MPS(D,d,D) x MPO(w,d,d,w) x environment(D,D,w)`, the
# dominant cost in every DMRG-style sweep (cost ~ O(D^3*d*w + D^2*d^2*w^2)). `sizes` is a list
# of bond dimensions `D` to sweep; physical dimension `d` and MPO bond `w` are held fixed
# (representative of a local spin/Hubbard-like model) since the literature identifies `D` as
# the dominant scaling knob.
#
# Also includes the 2-site "theta" tensor variant (`L - W - W - R` applied to a 2-site ket),
# which is what feeds the SVD/truncation step in 2-site DMRG.

const MPS_PHYS_DIM = 2   # d: physical dimension (spin-1/2)
const MPS_MPO_BOND = 6   # w: MPO bond dimension (local Hamiltonian)

function _mps_1site_case(D)
    d, w = MPS_PHYS_DIM, MPS_MPO_BOND
    # labels: 1=mpoL bond, 2=ket-L bond, 3=phys-in, 4=ket-R bond, 5=mpoR bond
    # output (negative): -10=out-L bond, -11=out-phys, -12=out-R bond
    indexlists = [
        [-10, 1, 2],      # L: (D, w, D)
        [2, 3, 4],        # ket: (D, d, D)
        [1, -11, 3, 5],   # MPO: (w, d, d, w)
        [4, 5, -12],       # R: (D, w, D)
    ]
    dims = Dict(1 => w, 2 => D, 3 => d, 4 => D, 5 => w, 10 => D, 11 => d, 12 => D)
    spec = NetworkSpec(indexlists, dims; output = [-10, -11, -12])
    return BenchmarkCase(:mps, "1site_D$(D)", (; D, d, w, variant = :onesite), spec)
end

function _mps_2site_case(D)
    d, w = MPS_PHYS_DIM, MPS_MPO_BOND
    # labels: 1=mpoL bond, 2=ket-L bond, 3=phys-in(1), 4=ket-mid bond, 5=phys-in(2),
    #         6=mpo-mid bond, 7=mpoR bond, 8=ket-R bond
    # output (negative): -10=out-L bond, -11=out-phys(1), -13=out-phys(2), -12=out-R bond
    indexlists = [
        [-10, 1, 2],       # L: (D, w, D)
        [2, 3, 4],         # ket1: (D, d, D)
        [4, 5, 8],         # ket2: (D, d, D)
        [1, -11, 3, 6],    # MPO1: (w, d, d, w)
        [6, -13, 5, 7],    # MPO2: (w, d, d, w)
        [8, 7, -12],        # R: (D, w, D)
    ]
    dims = Dict(1 => w, 2 => D, 3 => d, 4 => D, 5 => d, 6 => w, 7 => w, 8 => D, 10 => D, 11 => d, 12 => D, 13 => d)
    spec = NetworkSpec(indexlists, dims; output = [-10, -11, -13, -12])
    return BenchmarkCase(:mps, "2site_D$(D)", (; D, d, w, variant = :twosite), spec)
end

function _mps_cases(sizes)
    return vcat(
        BenchmarkCase[_mps_1site_case(D) for D in sizes],
        BenchmarkCase[_mps_2site_case(D) for D in sizes]
    )
end

register_category!(:mps, _mps_cases; sizes = (32, 64, 128, 256, 512))
