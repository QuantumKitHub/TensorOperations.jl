function tensorcontract_pb!(ΔC, C, ΔA, A, ΔB, B, α, β, pA, pB, pAB, conjA::Bool, conjB::Bool, ba...)
    ipAB = invperm(linearize(pAB))
    pdC = (
        TupleTools.getindices(ipAB, trivtuple(numout(pA))),
        TupleTools.getindices(ipAB, numout(pA) .+ trivtuple(numin(pB))),
    )
    ipA = (invperm(linearize(pA)), ())
    ipB = (invperm(linearize(pB)), ())
    conjΔC = conjA
    conjB′ = conjA ? conjB : !conjB
    tensorcontract!(
        ΔA,
        ΔC, pdC, conjΔC,
        B, reverse(pB), conjB′,
        ipA,
        conjA ? α : conj(α), One(), ba...
    )
    conjΔC = conjB
    conjA′ = conjB ? conjA : !conjA
    tensorcontract!(
        ΔB,
        A, reverse(pA), conjA′,
        ΔC, pdC, conjΔC,
        ipB,
        conjB ? α : conj(α), One(), ba...
    )
    Δα = if _needs_tangent(Tα)
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
    Δβ = if _needs_tangent(Tβ)
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
    if β === Zero()
        scale!(ΔC, β)
    else
        scale!(ΔC, conj(β))
    end
    return ΔC, ΔA, ΔB, Δα, Δβ 
end

function tensorcontract_pb!(ΔC, C, ΔA, A, ΔB, B, α::Zero, β::Zero, args...)
    scale!(ΔC, zero(eltype(C)))
    return ntuple(i -> nothing, 5)
end
