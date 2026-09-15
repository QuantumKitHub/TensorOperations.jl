# The downstream extension point: a `AbstractProvider` supplies the tensor type, default
# element type, backend and allocator to use when executing a (backend-agnostic) spec. This is
# deliberately a minimal interface (four required-ish methods) so that a downstream package
# (e.g. a symmetric/block-sparse tensor package) can add its own provider without touching
# anything else in this package.

"""
    AbstractProvider

Supertype for the downstream extension point of the benchmark suite. A provider ties a spec
(pure index/dimension data) to an actual tensor type, backend, and allocator to run it with.

Required methods:
- `scalartype(provider)`: the provider's default element type.
- `randtensor(provider, labels, dims, T=scalartype(provider))`: build a random tensor with the
  given (ordered) `labels`/`dims` and element type `T`.

Optional methods (with sensible defaults):
- `backend(provider) = DefaultBackend()`
- `allocator(provider) = DefaultAllocator()`
- `label(provider) = string(nameof(typeof(provider)))`
- `supports(provider, category::Symbol) = true`
- `rng(provider) = Random.default_rng()`: the RNG `randtensor` should draw from. Override
  with a *stored, stateful* RNG seeded once at construction (as [`ArrayProvider`](@ref) does)
  to make repeated suite runs reproducible -- a fresh RNG re-seeded on every call would just
  make every tensor within a run identical, which isn't the same thing.
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

The reference provider: plain dense `Array`s of element type `T`, exercising
TensorOperations.jl's own backends (`StridedNative`, `StridedBLAS`, `cuTENSORBackend`, ...).
This is how TensorOperations.jl dogfoods its own suite.

Tensors are allocated via `TensorOperations.tensoralloc` (so they go through `allocator`, not
a bare `Array` constructor) and filled via `randn!(rng, ...)`. `rng` is stored (not
reconstructed per call), so it advances across successive `randtensor` calls within one suite
build -- but is seeded identically across separate `ArrayProvider()` constructions, so two runs
of the same suite see the same input data.
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
