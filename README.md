<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/QuantumKitHub/TensorOperations.jl/blob/master/docs/src/assets/logo-dark.svg">
    <img alt="TensorOperations.jl logo" src="https://github.com/QuantumKitHub/TensorOperations.jl/blob/master/docs/src/assets/logo.svg" width="150">
</picture>

# TensorOperations.jl

Fast tensor operations using a convenient Einstein index notation.

| **Documentation** | **Digital Object Identifier** | **Downloads** |
|:-----------------:|:-----------------------------:|:-------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][downloads-img]][downloads-url] |

| **Build Status** | **PkgEval** | **Coverage** | **Quality assurance** |
|:----------------:|:------------:|:------------:|:---------------------:|
| [![CI][ci-img]][ci-url] | [![PkgEval][pkgeval-img]][pkgeval-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://QuantumKitHub.github.io/TensorOperations.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://QuantumKitHub.github.io/TensorOperations.jl/latest

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.3245496.svg
[doi-url]: https://doi.org/10.5281/zenodo.3245496

[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FTensorOperations&query=total_requests&label=Downloads
[downloads-url]: http://juliapkgstats.com/pkg/TensorOperations

[ci-img]: https://github.com/QuantumKitHub/TensorOperations.jl/workflows/CI/badge.svg
[ci-url]:
  https://github.com/QuantumKitHub/TensorOperations.jl/actions?query=workflow%3ACI

[pkgeval-img]: https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/TensorOperations.svg
[pkgeval-url]: https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/TensorOperations.html

[codecov-img]:
  https://codecov.io/gh/QuantumKitHub/TensorOperations.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuantumKitHub/TensorOperations.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

## What's new in v5

- Support for cuTENSOR v2 and with that more recent versions of [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).

- Improved support for automatic differentiation using reverse-mode rules with [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).

- Improved and extended support for backends and allocation strategies, with in particular support for allocating temporary objects using [Bumper.jl](https://github.com/MasonProtter/Bumper.jl).

- Breaking changes to part of the interface to make it more sustainable for future improvements and extensions.
  
## What's new in v4

- The `@tensor` macro now accepts keyword arguments to facilitate a variety of options that help with debugging, contraction cost and backend selection.

- Experimental support for automatic differentiation has been added by adding reverse-mode chainrules.

- The interface for custom types has been changed and thoroughly documented, making it easier to know what to implement. This has as a consequence that more general element types of tensors are now also possible.

- There is a new interface to work with backends, to allow for dynamic switching between different implementations of the primitive tensor operations or between different strategies for allocating new tensor objects.

- The support for `CuArray` objects is moved to a package extension, to avoid unnecessary CUDA dependencies for Julia versions >= 1.9

- The cache for temporaries has been removed due to its inconsistent and intricate interplay with multithreading.
  However, the new feature of specifying custom allocation strategies can be used to experiment with novel cache-like behaviour in the future.

> **WARNING:** TensorOperations 4.0 contains several breaking changes and cannot generally be expected to be compatible with previous versions.

### Code example

TensorOperations.jl is mostly used through the `@tensor` macro which allows one
to express a given operation in terms of
[index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format,
a.k.a. [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
(using Einstein's summation convention).

```julia
using TensorOperations
α = randn()
A = randn(5, 5, 5, 5, 5, 5)
B = randn(5, 5, 5)
C = randn(5, 5, 5)
D = zeros(5, 5, 5)
@tensor begin
    D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end
```

In the second to last line, the result of the operation will be stored in the
preallocated array `D`, whereas the last line uses a different assignment
operator `:=` in order to define and allocate a new array `E` of the correct
size. The contents of `D` and `E` will be equal.

For more detailed information, please see the documentation.

## Citing

See [`CITATION.bib`](CITATION.bib) for the relevant reference(s).
