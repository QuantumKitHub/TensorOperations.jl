# Based on the design of the GPUArrays.jl and MatrixAlgebraKit.jl test suites

"""
    TestSuite

Suite of tests for the primitive operations `tensoradd!`, `tensortrace!` and `tensorcontract!`
(and their allocating counterparts) of a TensorOperations backend.

Every result is compared to that of a reference backend, `BaseCopy()` by default, which is run
on CPU copies of the same inputs. Backends acting on other array types, such as GPU arrays, are
therefore checked against CPU results. `BaseCopy` itself is checked against straightforward
loop implementations through `TestSuite.LoopReference()`.

Packages providing a backend can include this test suite as follows:

```julia
using TensorOperations
include(joinpath(pkgdir(TensorOperations), "test", "testsuite", "TestSuite.jl"))
TestSuite.test_backend(MyBackend(); arraytype = MyArray)
```

See [`test_backend`](@ref) for the options to describe what the backend supports.
"""
module TestSuite

using Test
using TensorOperations
using TensorOperations: AbstractBackend, BaseCopy, DefaultAllocator, Index2Tuple, IndexError,
    linearize, numind
using LinearAlgebra: Diagonal
using Adapt: adapt
using Random
using VectorInterface: One, Zero

const tests = Dict{String, Function}()

"""
    @testsuite name (backend, options) -> begin ... end

Register the test function under `name`, which is then included in [`test_backend`](@ref) and
can also be called separately as `test_<name>(backend; kwargs...)`.
"""
macro testsuite(name, ex)
    safe_name = lowercase(replace(name, " " => "_", "/" => "_", "!" => ""))
    fn = Symbol("test_", safe_name)
    return quote
        $(esc(fn))(backend; kwargs...) = $(esc(ex))(backend, Options(; kwargs...))
        @assert !haskey(tests, $name) "testsuite already exists"
        tests[$name] = $(esc(fn))
    end
end

"""
    Options(; kwargs...)

Description of the backend under test, which determines the tests that are run. See the
fields for the keyword arguments and their defaults.
"""
Base.@kwdef struct Options
    "Type of arrays the backend acts on, e.g. `CuArray`; the inputs are moved to it with `Adapt.adapt`."
    arraytype::Type = Array
    "Element types to test."
    eltypes::Tuple = (Float32, Float64, ComplexF32, ComplexF64)
    """
    Combinations `(TC, TA, TB)` of element types of the output and the inputs, to test
    arguments with different element types; `tensoradd!` and `tensortrace!` use `(TC, TA)`.
    Pass `()` for backends that require all element types to be equal.
    """
    mixed_eltypes::Tuple = (
        (ComplexF64, Float64, ComplexF64), (Float64, Float32, Float64),
        (ComplexF32, Float64, Float32), (ComplexF64, ComplexF32, Float64),
    )
    "Backend computing the expected results, on `Array`s."
    reference = BaseCopy()
    "Allocator passed to the backend."
    allocator = DefaultAllocator()
    "Support for non-contiguous (strided) views of `arraytype`."
    views::Bool = true
    "Support for views with negative strides."
    negative_strides::Bool = true
    "Support for `Adjoint` and `Transpose` inputs."
    wrappers::Bool = true
    "Support for arrays with dimensions of size zero."
    zero_size::Bool = true
    "Support for `Diagonal` arguments in contractions."
    diagonal::Bool = false
    "Whether the backend rejects outputs that alias an input with an `ArgumentError`."
    aliasing::Bool = false
end

"""
    test_backend(backend; kwargs...)

Run the entire test suite for `backend`. The keyword arguments describe the backend and its
capabilities, see [`Options`](@ref) for the available options.
"""
function test_backend(backend; kwargs...)
    return @testset "$name" for (name, test) in sort!(collect(tests); by = first)
        test(backend; kwargs...)
    end
end

const rng = Random.Xoshiro(1234)
seed_rng!(seed) = Random.seed!(rng, seed)

precision(::Type{T}) where {T <: Number} = sqrt(eps(real(float(T))))
precision(Ts::Type...) = maximum(precision, Ts)

randarray(::Type{T}, sz::Dims) where {T} = randn(rng, T, sz)
randarray(::Type{T}, sz::Tuple{}) where {T} = fill(randn(rng, T))

# fill the output with `NaN` if `β` is zero, to verify that it is not read
maybe_poison!(C, β) = iszero(β) ? fill!(C, convert(eltype(C), NaN)) : C

# non-contiguous views of a larger array, possibly with negative strides
function stridedview(::Type{T}, sz::Dims; reversed::Bool = false) where {T}
    parent = randarray(T, 2 .* sz .+ 1)
    inds = map(n -> reversed ? StepRange(2n, -2, 2) : StepRange(2, 2, 2n), sz)
    return view(parent, inds...)
end

# the different scalar factors, including `Zero()` and `One()`, a numeric zero `β` that also
# needs to be treated as a strong zero, and a zero `α`
function scalars(::Type{T}) where {T}
    α, β = randn(rng, T), randn(rng, T)
    return ((One(), Zero()), (One(), One()), (α, zero(T)), (α, β), (zero(T), β))
end

# Moving data between devices
# ---------------------------
"""
    todevice(arraytype, x)

Copy `x` to an array of type `arraytype`, preserving views and wrappers.
"""
todevice(AT::Type, x) = adapt(AT, deepcopy(x))

"""
    tocpu(x)

Copy the values of `x` to a new `Array`.
"""
tocpu(x::AbstractArray) = Array(adapt(Array, x))

"""
    check_inplace(f!, backend, options, C, args...; rtol)

Run `f!(C, args..., backend, allocator)` on copies of `C` and `args` on the device, and compare
the result to `f!` with the reference backend on CPU copies. Also verifies that `f!` returns
its output argument, and leaves its inputs untouched.
"""
function check_inplace(f!, backend, options::Options, C, args...; rtol::Real)
    Cref = f!(tocpu(C), map(tocpu, args)..., options.reference, DefaultAllocator())

    Cdev = todevice(options.arraytype, C)
    argsdev = map(x -> todevice(options.arraytype, x), args)
    Cout = f!(Cdev, argsdev..., backend, options.allocator)
    @test Cout === Cdev
    @test isapprox(tocpu(Cout), Cref; rtol)
    @test all(map((x, y) -> tocpu(x) == tocpu(y), argsdev, args))
    return nothing
end

"""
    check_allocating(f, backend, options, args...; rtol)

Run `f(args..., backend, allocator)` on copies of `args` on the device, and compare the
result to `f` with the reference backend on CPU copies. Also verifies that the result is
again an array of type `arraytype`.
"""
function check_allocating(f, backend, options::Options, args...; rtol::Real)
    Cref = f(map(tocpu, args)..., options.reference, DefaultAllocator())

    argsdev = map(x -> todevice(options.arraytype, x), args)
    Cout = f(argsdev..., backend, options.allocator)
    @test Cout isa options.arraytype
    @test eltype(Cout) == eltype(Cref)
    @test isapprox(tocpu(Cout), Cref; rtol)
    return nothing
end

# Reference implementation
# ------------------------
"""
    LoopReference()

Backend that implements the primitive operations as loops over all elements, directly
following their definitions. It is slow, but its correctness can be verified by inspection,
and serves to verify the `BaseCopy` backend.
"""
struct LoopReference <: AbstractBackend end

# Map `Zero()` and `One()` to the strong `Bool` versions
_scalar(x::Number) = x
_scalar(::Zero) = false
_scalar(::One) = true

function _update(c, x, α, β)
    α, β = _scalar(α), _scalar(β)
    return iszero(β) ? α * x : β * c + α * x
end

function TensorOperations.tensoradd!(
        C::AbstractArray, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number, ::LoopReference, allocator = DefaultAllocator()
    )
    op = conjA ? conj : identity
    p = linearize(pA)
    for I in CartesianIndices(C)
        # dimension p[k] of A becomes dimension k of C
        J = zeros(Int, ndims(A))
        for k in 1:ndims(C)
            J[p[k]] = I[k]
        end
        C[I] = _update(C[I], op(A[J...]), α, β)
    end
    return C
end

function TensorOperations.tensortrace!(
        C::AbstractArray, A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number, ::LoopReference, allocator = DefaultAllocator()
    )
    op = conjA ? conj : identity
    lp, (q1, q2) = linearize(p), q
    traced = CartesianIndices(map(i -> size(A, i), q1))
    for I in CartesianIndices(C)
        J = zeros(Int, ndims(A))
        for k in 1:ndims(C)
            J[lp[k]] = I[k]
        end
        x = zero(eltype(C))
        for K in traced
            for k in eachindex(q1)
                J[q1[k]] = J[q2[k]] = K[k]
            end
            x += op(A[J...])
        end
        C[I] = _update(C[I], x, α, β)
    end
    return C
end

function TensorOperations.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple, α::Number, β::Number,
        ::LoopReference, allocator = DefaultAllocator()
    )
    opA = conjA ? conj : identity
    opB = conjB ? conj : identity
    lpAB = linearize(pAB)
    nA = length(pA[1])
    contracted = CartesianIndices(map(i -> size(A, i), pA[2]))
    for I in CartesianIndices(C)
        # the open indices of A followed by those of B; open index lpAB[k] is dimension k of C
        O = zeros(Int, numind(pAB))
        for k in 1:ndims(C)
            O[lpAB[k]] = I[k]
        end
        JA, JB = zeros(Int, ndims(A)), zeros(Int, ndims(B))
        for k in eachindex(pA[1])
            JA[pA[1][k]] = O[k]
        end
        for k in eachindex(pB[2])
            JB[pB[2][k]] = O[nA + k]
        end
        x = zero(eltype(C))
        for K in contracted
            for k in eachindex(pA[2])
                JA[pA[2][k]] = JB[pB[1][k]] = K[k]
            end
            x += opA(A[JA...]) * opB(B[JB...])
        end
        C[I] = _update(C[I], x, α, β)
    end
    return C
end

include("tensoradd.jl")
include("tensortrace.jl")
include("tensorcontract.jl")

end
