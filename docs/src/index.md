# TensorOperations.jl

*Fast tensor operations using a convenient Einstein index notation.*

## Table of contents

```@contents
Pages = ["index.md", "man/indexnotation.md", "man/functions.md", "man/interface.md", "man/backends.md", "man/autodiff.md", "man/implementation.md", "man/precompilation.md"]
Depth = 4
```

## Installation

Install with the package manager, `pkg> add TensorOperations`.

## Package features

The TensorOperations.jl package is centered around the following features:

  - A macro [`@tensor`](@ref) for conveniently specifying tensor contractions and index
    permutations via Einstein's index notation convention. The index notation is analyzed at
    compile time and lowered into primitive tensor operations, namely (permuted) linear
    combinations and inner and outer contractions. The macro supports several keyword
    arguments to customize the lowering process, namely to insert additional checks that
    help with debugging, to specify contraction order or to automatically determine optimal
    contraction order for given costs (see next bullet), and finally, to select different
    backends to evaluate those primitive operations.
  - The ability to optimize pairwise contraction order in complicated tensor contraction
    networks according to the algorithm in [this
    paper](https://doi.org/10.1103/PhysRevE.90.033315), where custom (compile time) costs
    can be specified, either as a keyword to [`@tensor`](@ref) or using the
    [`@tensoropt`](@ref) macro (for expliciteness and backward compatibility). This
    optimization is performed at compile time, and the resulting contraction order is hard
    coded into the resulting expression. The similar macro `@tensoropt_verbose` provides
    more information on the optimization process.
  - A function `ncon` (for network contractor) for contracting a group of tensors (a.k.a. a
    tensor network), as well as a corresponding `@ncon` macro that simplifies and optimizes
    this slightly. Unlike the previous macros, `ncon` and `@ncon` do not analyze the
    contractions at compile time, thus allowing them to deal with dynamic networks or index
    specifications.
  - (Experimental) support for automatic differentiation by supplying chain rules for the
    different tensor operations using the `ChainRules.jl` interface.
  - The ability to support different tensor types by overloading a minimal interface of
    tensor operations, or to support different implementation backends for the same tensor
    type.
  - The ability to support different allocation strategies, e.g. in order to deal with the
    allocation of temporary tensor objects in evaluating tensor network contractions.
  - An efficient default implementation for Julia Base arrays that qualify as strided, i.e.
    such that its entries are layed out according to a regular pattern in memory. The only
    exceptions are `ReinterpretedArray` objects. Additionally, `Diagonal` objects whose
    underlying diagonal data is stored as a strided vector are supported. This facilitates
    tensor contractions where one of the operands is e.g. a diagonal matrix of singular
    values or eigenvalues, which are returned as a `Vector` by Julia's `eigen` or `svd`
    method. This implementation for `AbstractArray` objects is based on
    [Strided.jl](https://github.com/Jutho/Strided.jl) for efficient (cache-friendly and
    multithreaded) tensor permutations (transpositions) and `gemm` from BLAS for
    contractions. There is also a fallback contraction strategy that is natively built using
    Strided.jl, e.g. for scalar types which are not supported by BLAS.
  - Support for `CuArray` objects if used together with
    [CUDA.jl and cuTENSOR.jl](https://github.com/JuliaGPU/CUDA.jl), by relying on (and thus
    providing a high level interface into) NVidia's
    [cuTENSOR](https://developer.nvidia.com/cutensor) library.
  - Support for `ROCArray` objects if used together with
    [AMDGPU.jl and hipTENSOR](https://github.com/JuliaGPU/AMDGPU.jl), by relying on (and thus
    providing a high level interface into) AMD's
    [hipTENSOR](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html) library.
  - Fall-back support for `AbstractArray` instances which are not strided, by using a
    general implementation using only constructions from Julia's `Base` and `LinearAlgebra`
    modules (`reshape`, `permutedims`, `mul!`, etc.) and some indexing in the case of
    partial traces (see below).

## Tensor operations

TensorOperations.jl supports 3 basic tensor operations, i.e. primitives in which every more
complicated tensor expression is deconstructed.

 1. **addition:** Add a (possibly scaled version of) one tensor to another tensor, where the
    indices of both arrays might appear in different orders. This operation combines normal
    tensor addition (or linear combination more generally) and index permutation. It
    includes as a special case copying one tensor into another with permuted indices.
 2. **trace or inner contraction:** Perform a trace/contraction over pairs of indices of a
    single tensor array, where the result is a lower-dimensional array.
 3. **(outer) contraction:** Perform a general contraction of two tensors, where some
    indices of one array are paired with corresponding indices in a second array.

## To do list

  - Make it easier to modify the contraction order algorithm or its cost function (e.g. to
    optimize based on memory footprint) or to splice in runtime information.
