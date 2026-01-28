#-------------------------------------------------------------------------------------------
# Operations
#-------------------------------------------------------------------------------------------

"""
    tensoradd!(C, A, pA, conjA, α=1, β=1 [, backend, allocator])

Compute `C = β * C + α * permutedims(opA(A), pA)`. The operation `opA` acts as `identity`
if `conjA` equals `:N` and as `conj` if `conjA` equals `:C`. Optionally specify a backend
implementation to use, and an allocator to be used if temporary tensor objects are needed.


!!! warning
    The permutation needs to be trivial or `C` must not be aliased with `A`.

See also [`tensoradd`](@ref).
"""
function tensoradd! end
# insert default α and β arguments
function tensoradd!(C, A, pA::Index2Tuple, conjA::Bool)
    return tensoradd!(C, A, pA, conjA, One(), One())
end
# insert default backend
function tensoradd!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number
    )
    return tensoradd!(C, A, pA, conjA, α, β, DefaultBackend())
end
# insert default allocator
function tensoradd!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend
    )
    return tensoradd!(C, A, pA, conjA, α, β, backend, DefaultAllocator())
end
# replace default backend with select_backend mechanism
function tensoradd!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend, allocator
    )
    if backend isa DefaultBackend
        newbackend = select_backend(tensoradd!, C, A)
        return tensoradd!(C, A, pA, conjA, α, β, newbackend, allocator)
    elseif backend isa NoBackend
        # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("No suitable backend found for tensoradd! and tensor types $TC and $TA"))
    else
        # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("Unknown backend $backend for tensoradd! and tensor types $TC and $TA"))
    end
end

"""
    tensortrace!(C, A, p, q, conjA, α=1, β=0 [, backend, allocator])

Compute `C = β * C + α * permutedims(opA(A), (p, q))`, where `A` is partially traced,
such that indices in `q[1]` are contracted with indices in `q[2]`, and the other indices
are permuted according to `p`. The operation `opA` acts as `identity` if `conjA` equals
`false` and as `conj` if `conjA` equals `true`. Optionally specify a backend implementation
to use, and an allocator to be used if temporary tensor objects are needed.


!!! warning
    The object `C` must not be aliased with `A`.

See also [`tensortrace`](@ref).
"""
function tensortrace! end
# insert default α and β arguments
function tensortrace!(C, A, p::Index2Tuple, q::Index2Tuple, conjA::Bool)
    return tensortrace!(C, A, p, q, conjA, One(), Zero())
end
# insert default backend
function tensortrace!(
        C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number
    )
    return tensortrace!(C, A, p, q, conjA, α, β, DefaultBackend())
end
# insert default allocator
function tensortrace!(
        C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number, backend
    )
    return tensortrace!(C, A, p, q, conjA, α, β, backend, DefaultAllocator())
end
# replace default backend with select_backend mechanism
function tensortrace!(
        C,
        A, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend, allocator
    )
    if backend isa DefaultBackend
        newbackend = select_backend(tensortrace!, C, A)
        return tensortrace!(C, A, p, q, conjA, α, β, newbackend, allocator)
    elseif backend isa NoBackend
        # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("No suitable backend found for tensortrace! and tensor types $TC and $TA"))
    else
        # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        throw(ArgumentError("Unknown backend $backend for tensortrace! and tensor types $TC and $TA"))
    end
end

"""
    tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α=1, β=0 [, backend, allocator])

Compute `C = β * C + α * permutedims(contract(opA(A), opB(B)), pAB)` without creating the
intermediate temporary, where `A` and `B` are contracted such that the indices `pA[2]` of
`A` are contracted with indices `pB[1]` of `B`. The remaining indices `(pA[1]..., pB[2]...)`
are then permuted according to `pAB`. The operation `opA` acts as `identity` if `conjA`
equals `false` and as `conj` if `conjA` equals `true`; the operation `opB` is determined by
`conjB` analogously. Optionally specify a backend implementation to use, and an allocator
to be used if temporary tensor objects are needed.

!!! warning 
    The object `C` must not be aliased with `A` or `B`.

See also [`tensorcontract`](@ref).
"""
function tensorcontract! end
# insert default α and β arguments
function tensorcontract!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple
    )
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, One(), Zero())
end
# insert default backend
function tensorcontract!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number
    )
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, DefaultBackend())
end
# insert default allocator
function tensorcontract!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number, backend
    )
    allocator = DefaultAllocator()
    return tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB, α, β, backend, allocator)
end
# replace default backend with select_backend mechanism
function tensorcontract!(
        C,
        A, pA::Index2Tuple, conjA::Bool,
        B, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend, allocator
    )
    if backend isa DefaultBackend
        newbackend = select_backend(tensorcontract!, C, A, B)
        return tensorcontract!(
            C, A, pA, conjA, B, pB, conjB, pAB, α, β, newbackend, allocator
        )
    elseif backend isa NoBackend
        # error for missing backend
        TC = typeof(C)
        TA = typeof(A)
        TB = typeof(B)
        throw(ArgumentError("No suitable backend found for tensorcontract! and tensor types $TC, $TA and $TB"))
    else
        # error for unknown backend
        TC = typeof(C)
        TA = typeof(A)
        TB = typeof(B)
        throw(ArgumentError("Unknown backend $backend for tensorcontract! and tensor types $TC, $TA and $TB"))
    end
end

"""
    tensorscalar(C)

Return the single element of a tensor-like object with zero indices or dimensions as
a value of the underlying scalar type.
"""
function tensorscalar end

#-------------------------------------------------------------------------------------------
# Allocations
#-------------------------------------------------------------------------------------------

"""
    tensorstructure(A, iA, conjA)

Obtain the information associated to indices `iA` of tensor `op(A)`, where `op` acts as
`identity` if `conjA` equals `false` and as `conj` if `conjA` equals `true`.
"""
function tensorstructure end

"""
    tensoradd_structure(A, pA, conjA)

Obtain the structure information of `C`, where `C` would be the output of
`tensoradd!(C, A, pA, conjA)`.
"""
function tensoradd_structure end

"""
    tensorcontract_structure(A, pA, conjA, B, pB, conjB, pAB)

Obtain the structure information of `C`, where `C` would be the output of
`tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)`.
"""
function tensorcontract_structure end

"""
    tensoradd_type(TC, A, pA, conjA)

Obtain the type information of `C`, where `C` would be the output of
`tensoradd!(C, A, pA, conjA)` with scalartype `TC`.
"""
function tensoradd_type end

"""
    tensorcontract_type(TC, A, pA, conjA, B, pB, conjB, pAB)

Obtain the type information of `C`, where `C` would be the output of
`tensorcontract!(C, A, pA, conjA, B, pB, conjB, pAB)` with scalar type `TC`.
"""
function tensorcontract_type end

"""
    tensoralloc(ttype, structure, [istemp=false, allocator])

Allocate memory for a tensor of type `ttype` and structure `structure`. The optional third
argument can be used to indicate that the result is used as a temporary tensor, for which
in some cases and with some allocators (the optional fourth argument) a different allocation
strategy might be used.

See also [`tensoralloc_add`](@ref), [`tensoralloc_contract`](@ref) and [`tensorfree!`](@ref).
"""
function tensoralloc end

"""
    tensorfree!(C, [allocator])

Provide a hint that the allocated memory of `C` can be released.

See also [`tensoralloc`](@ref).
"""
function tensorfree! end

"""
    allocator_checkpoint!(allocator) -> checkpoint

Mark a checkpoint for a given allocator.
This can for example be used to implement stack-based buffer schemes, such as provided by Bumper.jl.

See also [`allocator_reset!`](@ref).
"""
allocator_checkpoint!(allocator) = nothing

"""
    allocator_reset!(allocator, checkpoint)

Reset a given allocator to the state provided by the marked `checkpoint`.
This can for example be used to implement stack-based buffer schemes, such as provided by Bumper.jl.

See also [`allocator_checkpoint!`](@ref).
"""
allocator_reset!(allocator, checkpoint) = allocator

#-------------------------------------------------------------------------------------------
# Utility
#-------------------------------------------------------------------------------------------

"""
    tensorcost(A, i)
    
Compute the contraction cost associated with the `i`th index of a tensor, such that the
total cost of a pairwise contraction is found as the product of the costs of all contracted
indices and all uncontracted indices.
"""
function tensorcost end

"""
    checkcontractible(A, iA, conjA, B, iB, conjB, label)

Verify whether two tensors `opA(A)` and `opB(B)` are compatible for having their respective
index `iA` and `iB` contracted, and throws an error if not. The operation `opA` acts as
`identity` if `conjA` equals `false` and as `conj` if `conjA` equals `true`; the operation `opB`
is determined by `conjB` analogously.
"""
function checkcontractible end
