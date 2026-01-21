"""
    tensoradd_pullback!(ΔC, ΔA, C, A, pA::Index2Tuple, conjA::Bool, α, β, ba...) -> ΔC, ΔA, Δα, Δβ

Compute pullbacks for [`tensoradd!`](@ref), updating cotangent arrays and returning cotangent scalars.

See also [`pullback_dC`](@ref), [`tensoradd_pullback_dA`](@ref), [`tensoradd_pullback_dα`](@ref) and [`pullback_dβ`](@ref)
for computing pullbacks for the individual components.
"""
function tensoradd_pullback!(ΔC, ΔA, C, A, pA::Index2Tuple, conjA::Bool, α, β, ba...)
    dA = tensoradd_pullback_dA!(ΔA, ΔC, C, A, pA, conjA, α, ba...)
    dα = tensoradd_pullback_dα(ΔC, C, A, pA, conjA, α, ba...)
    dβ = pullback_dβ(ΔC, C, β)
    dC = pullback_dC!(ΔC, β)
    return dC, dA, dα, dβ
end

@doc """
    tensoradd_pullback_dA(ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)
    tensoradd_pullback_dA!(ΔA, ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)

Compute the pullback for [`tensoradd!`](@ref) with respect to the input `A`.
The mutating version can be used to accumulate the result into `ΔA`.

See also [`tensoradd_pullback_dA!`](@ref) for computing and updating the gradient in-place.
""" tensoradd_pullback_dA, tensoradd_pullback_dA!

function tensoradd_pullback_dA(ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)
    ipA = inversepermutation(pA, A)
    return tensorcopy(ΔC, ipA, conjA, conjA ? α : conj(α), ba...)
end
function tensoradd_pullback_dA!(ΔA, ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)
    if eltype(ΔC) <: Complex && eltype(ΔA) <: Real
        ΔAc = tensoradd_pullback_dA(ΔC, C, A, pA, conjA, α, ba...)
        ΔA .+= real.(ΔAc)
    else
        ipA = inversepermutation(pA, ΔA)
        tensoradd!(ΔA, ΔC, ipA, conjA, conjA ? α : conj(α), One(), ba...)
    end
    return ΔA
end

"""
    tensoradd_pullback_dα(ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)

Compute the pullback for [`tensoradd!]`(ref) with respect to scaling coefficient `α`.
"""
function tensoradd_pullback_dα(ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)
    _needs_tangent(α) || return nothing
    return tensorscalar(
        tensorcontract(
            A, repartition(pA, 0), !conjA,
            ΔC, trivialpermutation(numind(pA), 0), false,
            ((), ()), One(), ba...
        )
    )
end
