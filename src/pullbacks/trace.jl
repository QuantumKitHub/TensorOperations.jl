function tensortrace_pb!(ΔC, C, ΔA, A, α, β, p, q, conjA, ba...)
    ip = invperm((linearize(p)..., q[1]..., q[2]...))
    Es = map(q[1], q[2]) do i1, i2
        one(
            TensorOperations.tensoralloc_add(
                TensorOperations.scalartype(A), A, ((i1,), (i2,)), conjA
            )
        )
    end
    E = _kron(Es, ba)
    tensorproduct!(
        ΔA, ΔC, (trivtuple(numind(p)), ()), conjA,
        E, ((), trivtuple(numind(q))), conjA,
        (ip, ()),
        conjA ? α : conj(α), One(), ba...
    )
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
    if β === Zero()
        scale!(ΔC, β)
    else
        scale!(ΔC, conj(β))
    end
    return Δα, Δβ
end
