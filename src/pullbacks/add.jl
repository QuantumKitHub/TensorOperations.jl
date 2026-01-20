function tensoradd_pullback!(ΔC, ΔA, C, A, pA::Index2Tuple, conjA::Bool, α, β, ba...)
    ipA = invperm(linearize(pA))
    ΔAc = eltype(ΔC) <: Complex && eltype(ΔA) <: Real ? zerovector(A, VectorInterface.promote_add(ΔC, α)) : ΔA
    tensoradd!(ΔAc, ΔC, (ipA, ()), conjA, conjA ? α : conj(α), One(), ba...)
    if eltype(ΔC) <: Complex && eltype(ΔA) <: Real
        ΔA .+= real.(ΔAc)
    end
    Δα = if _needs_tangent(α)
        tensorscalar(
            tensorcontract(
                A, ((), linearize(pA)), !conjA,
                ΔC, trivialpermutation(numind(pA), 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    Δβ = if _needs_tangent(β)
        tensorscalar(
            tensorcontract(
                C, trivialpermutation(0, numind(pA)), true,
                ΔC, trivialpermutation(numind(pA), 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    scale!(ΔC, conj(β))
    return ΔC, ΔA, Δα, Δβ
end
