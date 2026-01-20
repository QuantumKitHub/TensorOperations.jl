function tensorcontract_pullback!(ΔC, ΔA, ΔB, C, A, B, α, β, pA, pB, pAB, conjA::Bool, conjB::Bool, ba...)
    ipAB = invperm(linearize(pAB))
    pdC = (
        TupleTools.getindices(ipAB, trivtuple(numout(pA))),
        TupleTools.getindices(ipAB, numout(pA) .+ trivtuple(numin(pB))),
    )
    ipA = (invperm(linearize(pA)), ())
    ipB = (invperm(linearize(pB)), ())
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
                C_αβ, ((), trivtuple(numind(pAB))), true,
                ΔC, (trivtuple(numind(pAB)), ()), false,
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
                C, ((), trivtuple(numind(pAB))), true,
                ΔC, (trivtuple(numind(pAB)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    scale!(ΔC, conj(β))
    return ΔC, ΔA, ΔB, Δα, Δβ
end
