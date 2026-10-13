"""
    AbstractProvider

The downstream extension point: ties a spec (pure index/dimension data) to an actual tensor
type, backend, and allocator.

Required: `scalartype(provider)`, `randtensor(provider, labels, dims, T=scalartype(provider))`.
Optional: `backend`, `allocator`, `label`, `supports(provider, category)`, `rng` (defaults
below) -- override `rng` with a *stored, stateful* RNG (as [`ArrayProvider`](@ref) does) so
repeated suite runs are reproducible without every tensor in one run being identical.
"""
abstract type AbstractProvider end

function TensorOperations.scalartype(p::AbstractProvider)
    error("`scalartype` not implemented for provider $(typeof(p))")
end

"""
    randtensor(provider, labels, dims, T=scalartype(provider))

Instantiate a random tensor with index order `labels` (a `Vector{Symbol}` or `Vector{Int}`,
only used for bookkeeping), size `dims` (matching `labels` elementwise), and element type `T`.
"""
function randtensor end

backend(p::AbstractProvider) = DefaultBackend()
allocator(p::AbstractProvider) = DefaultAllocator()
label(p::AbstractProvider) = string(nameof(typeof(p)))
supports(p::AbstractProvider, category::Symbol) = true
rng(p::AbstractProvider) = Random.default_rng()

"""
    ArrayProvider{T}(; backend=DefaultBackend(), allocator=DefaultAllocator(), rng=Random.Xoshiro(0x5eed5eed5eed5eed))

The reference provider: plain dense `Array{T}`, exercising TensorOperations.jl's own backends.
Allocates via `TensorOperations.tensoralloc` (goes through `allocator`) and fills via
`randn!(rng, ...)`; `rng` is stored, not reseeded per call, so separate `ArrayProvider()`s
reproduce the same data while tensors within one run still differ.
"""
struct ArrayProvider{T, B <: AbstractBackend, A, R <: Random.AbstractRNG} <: AbstractProvider
    backend::B
    allocator::A
    rng::R
end
function ArrayProvider{T}(;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator(),
        rng::Random.AbstractRNG = Random.Xoshiro(0x5eed5eed5eed5eed)
    ) where {T}
    return ArrayProvider{T, typeof(backend), typeof(allocator), typeof(rng)}(backend, allocator, rng)
end

TensorOperations.scalartype(::ArrayProvider{T}) where {T} = T
function randtensor(p::ArrayProvider, labels, dims, T = TensorOperations.scalartype(p))
    C = TensorOperations.tensoralloc(Array{T, length(dims)}, dims, Val(false), allocator(p))
    return randn!(rng(p), C)
end
backend(p::ArrayProvider) = p.backend
allocator(p::ArrayProvider) = p.allocator
label(p::ArrayProvider{T}) where {T} = "$(nameof(typeof(p.backend)))/$T"
rng(p::ArrayProvider) = p.rng
