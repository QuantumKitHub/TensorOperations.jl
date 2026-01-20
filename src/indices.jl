"""
    struct IndexError{<:AbstractString} <: Exception
    
Exception type for reporting errors in the index specification.
"""
struct IndexError{S <: AbstractString} <: Exception
    msg::S
end

const IndexTuple{N} = NTuple{N, Int}

"""
    Index2Tuple{N₁,N₂} = Tuple{NTuple{N₁,Int},NTuple{N₂,Int}}

A specification of a permutation of `N₁ + N₂` indices that are partitioned into `N₁` left
and `N₂` right indices.
"""
const Index2Tuple{N₁, N₂} = Tuple{IndexTuple{N₁}, IndexTuple{N₂}}

linearize(p::Index2Tuple) = (p[1]..., p[2]...)
numout(p::Index2Tuple) = length(p[1])
numin(p::Index2Tuple) = length(p[2])
numind(p::Index2Tuple) = numout(p) + numin(p)

trivialpermutation(p::IndexTuple{N}) where {N} = ntuple(identity, Val(N))
function trivialpermutation(p::Index2Tuple)
    return (trivialpermutation(p[1]), numout(p) .+ trivialpermutation(p[2]))
end
trivialpermutation(N::Integer) = ntuple(identity, N)
trivialpermutation(N₁::Integer, N₂::Integer) = (trivialpermutation(N₁), trivialpermutation(N₂) .+ N₁)

istrivialpermutation(p::IndexTuple) = p == trivialpermutation(p)
istrivialpermutation(p::Index2Tuple) = p == trivialpermutation(p)

Base.@constprop :aggressive function repartition(p::IndexTuple, N₁::Int)
    length(p) >= N₁ || throw(ArgumentError("cannot repartition $(typeof(p)) to $N₁, $(length(p) - N₁)"))
    partition = trivialpermutation(N₁, length(p) - N₁)
    return TupleTools.getindices(p, partition[1]), TupleTools.getindices(p, partition[2])
end
@inline repartition(p::Index2Tuple, N₁::Int) = repartition(linearize(p), N₁)

@inline repartition(p::Union{IndexTuple, Index2Tuple}, ::Index2Tuple{N₁}) where {N₁} =
    repartition(p, N₁)
@inline repartition(p::Union{IndexTuple, Index2Tuple}, A::AbstractArray) = repartition(p, ndims(A))

"""
    inversepermutation(p::Index2Tuple) -> ip::IndexTuple
    inversepermutation(p::Index2Tuple, N₁::Int) -> ip::Index2Tuple
    inversepermutation(p::Index2Tuple, partition_as::Index2Tuple) -> ip::Index2Tuple
    inversepermutation(p::Index2Tuple, partition_as::AbstractArray) -> ip::Index2Tuple

Compute the inverse permutation associated to `p`.
If no extra arguments are provided, the result is returned as a single `IndexTuple`.
Otherwise, the extra arguments are used to partition the inverse permutation into an `Index2Tuple`.
"""
inversepermutation(p::Index2Tuple) = invperm(linearize(p))
function inversepermutation(p::Index2Tuple, args...)
    ip = invperm(linearize(p))
    return repartition(ip, args...)
end

"""
    const LabelType = Union{Int,Symbol,Char}

Alias for supported label types.
"""
const LabelType = Union{Int, Symbol, Char}

"""
    const Labels{I<:LabelType} = Union{Tuple{Vararg{I}},Vector{I}}

Alias for supported label containers.
"""
const Labels{I <: LabelType} = Union{Tuple{Vararg{I}}, AbstractVector{I}}
