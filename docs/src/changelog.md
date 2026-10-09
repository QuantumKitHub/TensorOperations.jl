# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/QuantumKitHub/TensorOperations.jl/compare/v5.8.2...HEAD)

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Performance

- `ncon` partitions its intermediates as `(open indices of A; open indices of B)` instead of placing all indices in the codomain, which avoids repartitioning copies for tensor types such as `TensorMap`s with symmetries ([#311](https://github.com/QuantumKitHub/TensorOperations.jl/issues/311)).

## [5.8.2](https://github.com/QuantumKitHub/TensorOperations.jl/compare/v5.8.1...v5.8.2) - 2026-09-30

### Changed

- Compat with VectorInterface is extended to include 0.7 ([#306](https://github.com/QuantumKitHub/TensorOperations.jl/pull/306)).

### Fixed

- `dβ` in the Enzyme rule for `tensoradd!` when an `Active` `β` is zero: `C` was cached based on the value of `β` instead of its activity ([#307](https://github.com/QuantumKitHub/TensorOperations.jl/pull/307)).
- The Enzyme rules no longer write into a shadow that aliases the primal (`dval === val`) under runtime activity, and skip the unneeded copy of `C` ([#305](https://github.com/QuantumKitHub/TensorOperations.jl/pull/305)).

### Performance

- The ChainRules rules no longer copy or retain `C` when `β = Zero()` ([#308](https://github.com/QuantumKitHub/TensorOperations.jl/pull/308)).

## [5.8.1](https://github.com/QuantumKitHub/TensorOperations.jl/compare/v5.8.0...v5.8.1) - 2026-09-15

### Added

- `tensorfree!` for GPU arrays with `DefaultAllocator` ([#300](https://github.com/QuantumKitHub/TensorOperations.jl/pull/300)).

## [5.8.0](https://github.com/QuantumKitHub/TensorOperations.jl/compare/v5.7.0...v5.8.0) - 2026-08-13

### Added

- `TBLISBackend`, an opt-in backend routing `tensoradd!`, `tensortrace!` and `tensorcontract!` through [TBLIS.jl](https://github.com/QuantumKitHub/TBLIS.jl), available as a package extension. This supersedes the standalone TensorOperationsTBLIS.jl ([#290](https://github.com/QuantumKitHub/TensorOperations.jl/pull/290)).
- Buffer-backed allocators for GPU array types, which serve temporaries from a single preallocated device buffer: `CUDABufferAllocator`, `AMDBufferAllocator` and `JLBufferAllocator` ([#293](https://github.com/QuantumKitHub/TensorOperations.jl/pull/293), [#295](https://github.com/QuantumKitHub/TensorOperations.jl/pull/295), [#296](https://github.com/QuantumKitHub/TensorOperations.jl/pull/296)).

### Fixed

- Plan leak in the cuTENSOR implementation of `tensortrace!`, which retained the reduction workspace until finalization ([#294](https://github.com/QuantumKitHub/TensorOperations.jl/pull/294)).

### Performance

- `tensoradd!` bypasses `PermutedDimsArray` for `Diagonal` arguments ([#297](https://github.com/QuantumKitHub/TensorOperations.jl/pull/297)).
