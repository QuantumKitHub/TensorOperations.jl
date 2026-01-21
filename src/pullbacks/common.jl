# To avoid computing rrules for α and β when these aren't needed, we want to have a
# type-stable quick bail-out
"""
    _needs_tangent(x)
    _needs_tangent(::Type{T})

Determine whether a value requires tangent computation during automatic differentiation.
Returns `false` for constants like Integer, One, and Zero types to avoid unnecessary computation
in automatic differentiation. Returns `true` only for general Numbers.
"""
_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{<:Number}) = true
_needs_tangent(::Type{<:Integer}) = false
_needs_tangent(::Type{<:Union{One, Zero}}) = false
_needs_tangent(::Type{Complex{T}}) where {T} = _needs_tangent(T)

# (partial) pullbacks that are shared
@doc """
    pullback_dC(ΔC, β)
    pullback_dC!(ΔC, β)

For functions of the form `f!(C, β, ...) = βC + ...`, compute the pullback with respect to `C`.
""" pullback_dC, pullback_dC!

pullback_dC!(ΔC, β) = scale!(ΔC, conj(β))
pullback_dC(ΔC, β) = scale(ΔC, conj(β))

@doc """
    pullback_dβ(ΔC, C, β)

For functions of the form `f!(C, β, ...) = βC + ...`, compute the pullback with respect to `β`.
""" pullback_dβ

pullback_dβ(ΔC, C, β) = _needs_tangent(β) ? inner(C, ΔC) : nothing
