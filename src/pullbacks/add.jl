function tensoradd_pullback!(ΔC, ΔA, C, A, pA::Index2Tuple, conjA::Bool, α, β, ba...)
    dA = tensoradd_pullback_dA!(ΔA, ΔC, C, A, pA, conjA, α, ba...)
    dα = tensoradd_pullback_dα(ΔC, C, A, pA, conjA, α, ba...)
    dβ = pullback_dβ(ΔC, C, β)
    dC = pullback_dC!(ΔC, β)
    return dC, dA, dα, dβ
end

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

function tensoradd_pullback_dα(ΔC, C, A, pA::Index2Tuple, conjA::Bool, α, ba...)
    return if _needs_tangent(α)
        tensorscalar(
            tensorcontract(
                A, repartition(pA, 0), !conjA,
                ΔC, trivialpermutation(numind(pA), 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
end
