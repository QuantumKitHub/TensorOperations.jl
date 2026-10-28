using TensorOperations
using DynamicPolynomials
using Test

using VectorInterface: VectorInterface
const PolyTypes = Union{<:AbstractPolynomialLike, <:AbstractTermLike, <:AbstractMonomialLike}
# not clear if this is really the true `scalartype` we want
VectorInterface.scalartype(T::Type{<:PolyTypes}) = T
VectorInterface.zerovector!!(v::PolyTypes) = zero(v)
function VectorInterface.add!!(w::PolyTypes, v::PolyTypes, α::Number, β::Number)
    return w * β + v * α
end
VectorInterface.scale!!(v::PolyTypes, α::Number) = v * α

@polyvar a[1:2, 1:2] b[1:2, 1:2]
@tensor c[i, k] := a[i, j] * b[j, k]
@tensor d[i, k] := a[i, j] * b[j, k] + 2 * a[i, k]
@test c == a * b
@test d == a * b + 2 * a
