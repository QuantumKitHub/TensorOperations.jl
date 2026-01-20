_kron(Es::NTuple{1}, ba) = Es[1]
function _kron(Es::NTuple{N, Any}, ba) where {N}
    E1 = Es[1]
    E2 = _kron(Base.tail(Es), ba)
    p2 = trivialpermutation(0, 2N - 2)
    p = ((1, (2 .+ trivialpermutation(N - 1))...), (2, ((N + 1) .+ trivialpermutation(N - 1))...))
    return tensorproduct(p, E1, ((1, 2), ()), false, E2, p2, false, One(), ba...)
end

# To avoid computing rrules for α and β when these aren't needed, we want to have a
# type-stable quick bail-out
_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{<:Number}) = true
_needs_tangent(::Type{<:Integer}) = false
_needs_tangent(::Type{<:Union{One, Zero}}) = false
