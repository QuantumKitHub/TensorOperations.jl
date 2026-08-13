# Backends for tensor operations
#--------------------------------
"""
    abstract type AbstractBackend
    
Abstract supertype for all backends that can be used for tensor operations. In particular,
these control different implementations of executing the basic operations.
"""
abstract type AbstractBackend end

"""
    DefaultBackend()

Default backend for tensor operations if no explicit backend is specified. This will select
an actual implementation backend using the `select_backend(tensorfun, tensors...)` mechanism.
"""
struct DefaultBackend <: AbstractBackend end

"""
    NoBackend()

Backend that will be returned if no suitable backend can be found for the given tensors.
"""
struct NoBackend <: AbstractBackend end

"""
    select_backend([tensorfun::Function], tensors...) -> AbstractBackend

Select the default backend for the given tensors or tensortypes. If `tensorfun` is provided,
it is possible to more finely control the backend selection based on the function as well.
"""
select_backend(tensorfun::Function, tensors...) = select_backend(tensors...)
select_backend(tensors...) = NoBackend()

# Base backends
#-----------------
"""
    BaseCopy()

Backend for tensor operations that should work for all `AbstractArray` types and only uses
functions from the `Base` module, as well as `LinearAlgebra.mul!`.
"""
struct BaseCopy <: AbstractBackend end

"""
    BaseView()

Backend for tensor operations that should work for all `AbstractArray` types and only uses
functions from the `Base` module, as well as `LinearAlgebra.mul!`, and furthermore tries to
avoid any intermediate allocations by using views.
"""
struct BaseView <: AbstractBackend end

# Strided backends
#-----------------
"""
    StridedNative()
    
Backend for tensor operations that is based on `StridedView` objects with native Julia
implementations of tensor operations.
"""
struct StridedNative <: AbstractBackend end

"""
    StridedBLAS()
    
Backend for tensor operations that is based on using `StridedView` objects and rephrasing
the tensor operations as BLAS operations.
"""
struct StridedBLAS <: AbstractBackend end

const StridedBackend = Union{StridedNative, StridedBLAS}

# TBLIS backend
#--------------
"""
    TBLISBackend()

Backend for tensor operations on strided arrays that is based on the [TBLIS](https://github.com/devinamatthews/tblis) library.
TBLIS performs tensor additions, traces and contractions directly on strided memory, without the transpositions and temporaries that a BLAS-based approach requires.
This backend is only available through a package extension for [TBLIS.jl](https://github.com/QuantumKitHub/TBLIS.jl).

TBLIS requires all tensors in a single operation to be of the same element type, and only supports `BLASFloat`s.

!!! warning
    This backend is currently not supported on Windows: the `tblis_jll` binaries for that platform
    abort the process from inside the library, so calls are rejected with an `ArgumentError`.

!!! warning
    As of `tblis_jll` v1.3, TBLIS has no competitive support for complex element types: its
    complex contraction kernels run an order of magnitude slower than the corresponding BLAS calls.
    Prefer [`StridedBLAS`](@ref TensorOperations.StridedBLAS) for complex-valued contractions.
"""
struct TBLISBackend <: AbstractBackend end

# CuTENSOR backend
#-----------------
"""
    cuTENSORBackend()
Backend for tensor operations that is based on the NVIDIA cuTENSOR library.
"""
struct cuTENSORBackend <: AbstractBackend end
