function tensortrace_pullback!(ΔC, ΔA, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α, β, ba...)
    ip = repartition(invperm((linearize(p)..., linearize(q)...)), numind(p) + numind(q))
    Es = map(q[1], q[2]) do i1, i2
        one(
            TensorOperations.tensoralloc_add(
                TensorOperations.scalartype(A), A, ((i1,), (i2,)), conjA
            )
        )
    end
    E = _kron(Es, ba)
    ΔAc = eltype(ΔC) <: Complex && eltype(ΔA) <: Real ? zerovector(A, VectorInterface.promote_add(ΔC, α)) : ΔA
    tensorproduct!(
        ΔAc, ΔC, trivialpermutation(numind(p), 0), conjA,
        E, trivialpermutation(0, numind(q)), conjA,
        (ip, ()),
        conjA ? α : conj(α), One(), ba...
    )
    if eltype(ΔC) <: Complex && eltype(ΔA) <: Real
        ΔA .+= real.(ΔAc)
    end
    C_αβ = tensortrace(A, p, q, false, One(), ba...)
    Δα = if _needs_tangent(α)
        tensorscalar(
            tensorcontract(
                C_αβ, trivialpermutation(0, numind(p)),
                !conjA,
                ΔC, trivialpermutation(numind(p), 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    Δβ = if _needs_tangent(β)
        tensorscalar(
            tensorcontract(
                C, trivialpermtation(0, numind(p)), true,
                ΔC, trivialpermutation(numind(p), 0), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    scale!(ΔC, conj(β))
    return ΔC, ΔA, Δα, Δβ
end
