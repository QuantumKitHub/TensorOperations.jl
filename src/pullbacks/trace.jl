"""
    tensortrace_pullback!(ΔC, ΔA, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α::Number, β::Number, ba...) -> ΔC, ΔA, Δα, Δβ

Compute pullbacks for [`tensortrace!`](@ref), updating cotangent arrays and returning cotangent scalars.

See also [`pullback_dC`](@ref), [`tensortrace_pullback_dA`](@ref), [`tensortrace_pullback_dα`](@ref) and [`pullback_dβ`](@ref)
for computing pullbacks for the individual components.
"""
function tensortrace_pullback!(
        ΔC, ΔA,
        C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number, ba...
    )
    dA = tensortrace_pullback_dA!(ΔA, ΔC, C, A, p, q, conjA, α, ba...)
    dα = tensortrace_pullback_dα(ΔC, C, A, p, q, conjA, α, ba...)
    dβ = pullback_dβ(ΔC, C, β)
    dC = pullback_dC!(ΔC, β)

    return dC, dA, dα, dβ
end

@doc """
    tensortrace_pullback_dA(ΔC, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α::Number, ba...)
    tensortrace_pullback_dA!(ΔA, ΔC, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α::Number, ba...)

Compute the pullback for [`tensortrace!`](@ref) with respect to the input `A`.
The mutating version can be used to accumulate the result into `ΔA`.
""" tensortrace_pullback_dA, tensortrace_pullback_dA!

function tensortrace_pullback_dA(
        ΔC, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, ba...
    )
    ip = repartition(invperm((linearize(p)..., linearize(q)...)), A)
    Es = map(q[1], q[2]) do i1, i2
        one(
            TensorOperations.tensoralloc_add(
                TensorOperations.scalartype(A), A, ((i1,), (i2,)), conjA
            )
        )
    end
    E = _kron(Es, ba)

    return tensorproduct(
        ΔC, trivialpermutation(numind(p), 0), conjA,
        E, trivialpermutation(0, numind(q)), conjA,
        ip,
        conjA ? α : conj(α), ba...
    )
end
function tensortrace_pullback_dA!(
        ΔA, ΔC, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, ba...
    )

    if eltype(ΔC) <: Complex && eltype(ΔA) <: Real
        ΔAc = tensortrace_pullback_dA(ΔC, C, A, p, q, conjA, α, ba...)
        ΔA .+= real.(ΔAc)
    else
        ip = repartition(invperm((linearize(p)..., linearize(q)...)), A)
        Es = map(q[1], q[2]) do i1, i2
            one(
                TensorOperations.tensoralloc_add(
                    TensorOperations.scalartype(A), A, ((i1,), (i2,)), conjA
                )
            )
        end
        E = _kron(Es, ba)
        tensorproduct!(
            ΔA, ΔC, trivialpermutation(numind(p), 0), conjA,
            E, trivialpermutation(0, numind(q)), conjA,
            ip,
            conjA ? α : conj(α), One(), ba...
        )
    end

    return ΔA
end

"""
    tensortrace_pullback_dα(ΔC, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool, α::Number, ba...)

Compute the pullback for [`tensortrace!`](@ref) with respect to scaling coefficient `α`.
"""
function tensortrace_pullback_dα(
        ΔC, C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, ba...
    )
    return if _needs_tangent(α)
        C_αβ = tensortrace(A, p, q, false, One(), ba...)
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
end
