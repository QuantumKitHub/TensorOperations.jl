function tensortrace_pullback!(ΔC, ΔA, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α, β, ba...)
    ip = invperm((linearize(p)..., q[1]..., q[2]...))
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
        ΔAc, ΔC, (trivtuple(numind(p)), ()), conjA,
        E, ((), trivtuple(numind(q))), conjA,
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
                C_αβ, ((), trivtuple(numind(p))),
                !conjA,
                ΔC, (trivtuple(numind(p)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    Δβ = if _needs_tangent(β)
        tensorscalar(
            tensorcontract(
                C, ((), trivtuple(numind(p))), true,
                ΔC, (trivtuple(numind(p)), ()), false,
                ((), ()), One(), ba...
            )
        )
    else
        nothing
    end
    scale!(ΔC, conj(β))
    return ΔC, ΔA, Δα, Δβ
end
