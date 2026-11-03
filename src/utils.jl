_kron(Es::NTuple{1}, ba) = Es[1]
function _kron(Es::NTuple{N, Any}, ba) where {N}
    E1 = Es[1]
    E2 = _kron(Base.tail(Es), ba)
    p2 = trivialpermutation(0, 2N - 2)
    p = (
        (1, (2 .+ trivialpermutation(N - 1))...),
        (2, ((N + 1) .+ trivialpermutation(N - 1))...),
    )
    return tensorproduct(E1, ((1, 2), ()), false, E2, p2, false, p, One(), ba...)
end

# `C = β * C`, for operations without anything to add to `C`, such as traces or contractions
# over dimensions of size zero; `β = 0` overwrites `C`, which may contain `NaN`s
_scale!(C::AbstractArray, β::Number) = iszero(β) ? fill!(C, zero(eltype(C))) : (C .= β .* C)
