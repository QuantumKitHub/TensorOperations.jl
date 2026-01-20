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
                ΔC, (trivtuple(numind(pA)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    Δβ = if _needs_tangent(β)
        tensorscalar(
            tensorcontract(
                C, ((), trivtuple(numind(pA))), true,
                ΔC, (trivtuple(numind(pA)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    scale!(ΔC, conj(β))
    return ΔC, ΔA, Δα, Δβ
end
