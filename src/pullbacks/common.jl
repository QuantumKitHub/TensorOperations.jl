# To avoid computing rrules for α and β when these aren't needed, we want to have a
# type-stable quick bail-out
_needs_tangent(x) = _needs_tangent(typeof(x))
_needs_tangent(::Type{<:Number}) = true
_needs_tangent(::Type{<:Integer}) = false
_needs_tangent(::Type{<:Union{One, Zero}}) = false

# (partial) pullbacks that are shared
pullback_dC!(ΔC, β) = scale!(ΔC, conj(β))
pullback_dC(ΔC, β) = scale(ΔC, conj(β))
pullback_dβ(ΔC, C, β) = _needs_tangent(β) ? inner(C, ΔC) : nothing
