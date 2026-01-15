function tensoradd_pb!(ΔC, C, ΔA, A, α, β, pA, conjA::Bool, ba...)
    ipA = invperm(linearize(pA))
    tensoradd!(ΔA, ΔC, (ipA, ()), conjA, conjA ? α : conj(α), One(), ba...)
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
    if β === Zero()
        scale!(ΔC, β)
    else
        scale!(ΔC, conj(β))
    end
    return Δα, Δβ
end
