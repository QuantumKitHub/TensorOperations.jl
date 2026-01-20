function tensorcontract_pullback!(ΔC, ΔA, ΔB, C, A, pA::Index2Tuple, conjA::Bool, B, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple, α, β, ba...)
    pdC = inversepermutation(pAB, numout(pA))
    ipA = inversepermutation(pA, A)
    ipB = inversepermutation(pB, B)

    conjΔC = conjA
    conjB′ = conjA ? conjB : !conjB
    ΔAc = eltype(ΔC) <: Complex && eltype(ΔA) <: Real ? zerovector(A, VectorInterface.promote_add(ΔC, α)) : ΔA
    tensorcontract!(
        ΔAc,
        ΔC, pdC, conjΔC,
        B, reverse(pB), conjB′,
        ipA,
        conjA ? α : conj(α), One(), ba...
    )
    if eltype(ΔC) <: Complex && eltype(ΔA) <: Real
        ΔA .+= real.(ΔAc)
    end

    conjΔC = conjB
    conjA′ = conjB ? conjA : !conjA
    ΔBc = eltype(ΔC) <: Complex && eltype(ΔB) <: Real ? zerovector(B, VectorInterface.promote_add(ΔC, α)) : ΔB
    tensorcontract!(
        ΔBc,
        A, reverse(pA), conjA′,
        ΔC, pdC, conjΔC,
        ipB,
        conjB ? α : conj(α), One(), ba...
    )
    if eltype(ΔC) <: Complex && eltype(ΔB) <: Real
        ΔB .+= real.(ΔBc)
    end

    Δα = if _needs_tangent(α)
        C_αβ = tensorcontract(A, pA, conjA, B, pB, conjB, pAB, One(), ba...)
        # TODO: consider using `inner`
        tensorscalar(
            tensorcontract(
                C_αβ, trivialtuple(0, numind(pAB)), true,
                ΔC, trivialtuple(numind(pAB, 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end

    Δβ = if _needs_tangent(β)
        # TODO: consider using `inner`
        tensorscalar(
            tensorcontract(
                C, trivialtuple(0, numind(pAB)), true,
                ΔC, trivialtuple(numind(pAB), 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    scale!(ΔC, conj(β))
    return ΔC, ΔA, ΔB, Δα, Δβ
end
