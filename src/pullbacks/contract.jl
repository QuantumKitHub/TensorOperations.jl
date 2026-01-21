"""
    tensorcontract_pullback!(ΔC, ΔA, ΔB, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number, β::Number, ba...) -> ΔC, ΔA, ΔB, Δα, Δβ

Compute pullbacks for [`tensorcontract!`](@ref), updating cotangent arrays and returning cotangent scalars.

See also [`pullback_dC`](@ref), [`tensorcontract_pullback_dA`](@ref), [`tensorcontract_pullback_dB`](@ref),
[`tensorcontract_pullback_dα`](@ref) and [`pullback_dβ`](@ref) for computing pullbacks for the individual components.
"""
function tensorcontract_pullback!(
        ΔC, ΔA, ΔB,
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ba...
    )
    dA = tensorcontract_pullback_dA!(ΔA, ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...)
    dB = tensorcontract_pullback_dB!(ΔB, ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...)
    dα = tensorcontract_pullback_dα(ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...)
    dβ = pullback_dβ(ΔC, C, β)
    dC = pullback_dC!(ΔC, β)
    return dC, dA, dB, dα, dβ
end

@doc """
    tensorcontract_pullback_dA(ΔC, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number, ba...)
    tensorcontract_pullback_dA!(ΔA, ΔC, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number, ba...)

Compute the pullback for [`tensorcontract!`](@ref) with respect to the input `A`.
The mutating version can be used to accumulate the result into `ΔA`.

See also [`tensorcontract_pullback_dB`](@ref) and [`tensorcontract_pullback_dB!`](@ref) for the pullback with respect to `B`.
""" tensorcontract_pullback_dA, tensorcontract_pullback_dA!

function tensorcontract_pullback_dA(
        ΔC, C,
        A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, ba...
    )
    pdC = inversepermutation(pAB, numout(pA))
    ipA = inversepermutation(pA, A)
    return tensorcontract(
        ΔC, pdC, conjA, B, reverse(pB), conjA ? conjB : !conjB,
        ipA, conjA ? α : conj(α), ba...
    )
end
function tensorcontract_pullback_dA!(
        ΔA, ΔC, C,
        A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, ba...
    )

    if eltype(ΔC) <: Complex && eltype(ΔA) <: Real
        ΔAc = tensorcontract_pullback_dA(ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...)
        ΔA .+= real.(ΔAc)
    else
        pdC = inversepermutation(pAB, numout(pA))
        ipA = inversepermutation(pA, A)
        tensorcontract!(
            ΔA,
            ΔC, pdC, conjA, B, reverse(pB), conjA ? conjB : !conjB,
            ipA, conjA ? α : conj(α), One(), ba...
        )
    end

    return ΔA
end

@doc """
    tensorcontract_pullback_dB(ΔC, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number, ba...)
    tensorcontract_pullback_dB!(ΔB, ΔC, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number, ba...)

Compute the pullback for [`tensorcontract!`](@ref) with respect to the input `B`.
The mutating version can be used to accumulate the result into `ΔB`.

See also [`tensorcontract_pullback_dA`](@ref) and [`tensorcontract_pullback_dA!`](@ref) for the pullback with respect to `A`.
""" tensorcontract_pullback_dB, tensorcontract_pullback_dB!

function tensorcontract_pullback_dB(
        ΔC, C,
        A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, ba...
    )
    pdC = inversepermutation(pAB, numout(pA))
    ipB = inversepermutation(pB, B)
    return tensorcontract(
        A, reverse(pA), conjB ? conjA : !conjA, ΔC, pdC, conjB,
        ipB, conjB ? α : conj(α), ba...
    )
end
function tensorcontract_pullback_dB!(
        ΔB, ΔC, C,
        A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, ba...
    )
    if eltype(ΔC) <: Complex && eltype(ΔB) <: Real
        ΔBc = tensorcontract_pullback_dB(ΔC, C, A, pA, conjA, B, pB, conjB, pAB, α, ba...)
        ΔB .+= real.(ΔBc)
    else
        pdC = inversepermutation(pAB, numout(pA))
        ipB = inversepermutation(pB, B)
        tensorcontract!(
            ΔB,
            A, reverse(pA), conjB ? conjA : !conjA, ΔC, pdC, conjB,
            ipB, conjB ? α : conj(α), One(), ba...
        )
    end

    return ΔB
end

"""
    tensorcontract_pullback_dα(ΔC, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α::Number, ba...)

Compute the pullback for [`tensorcontract!`](@ref) with respect to scaling coefficient `α`.
"""
function tensorcontract_pullback_dα(
        ΔC, C,
        A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, ba...
    )
    return if _needs_tangent(α)
        C_αβ = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
        inner(C_αβ, ΔC)
    else
        nothing
    end
end
