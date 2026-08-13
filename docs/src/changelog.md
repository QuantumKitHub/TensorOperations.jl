# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/QuantumKitHub/TensorOperations.jl/compare/v5.8.0...HEAD)

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Performance

## [5.8.0](https://github.com/QuantumKitHub/TensorOperations.jl/compare/v5.7.0...v5.8.0) - 2026-08-13

### Added

- `TBLISBackend`, an opt-in backend routing `tensoradd!`, `tensortrace!` and `tensorcontract!` through [TBLIS.jl](https://github.com/QuantumKitHub/TBLIS.jl), available as a package extension. This supersedes the standalone TensorOperationsTBLIS.jl ([#290](https://github.com/QuantumKitHub/TensorOperations.jl/pull/290)).
- Buffer-backed allocators for GPU array types, which serve temporaries from a single preallocated device buffer: `CUDABufferAllocator`, `AMDBufferAllocator` and `JLBufferAllocator` ([#293](https://github.com/QuantumKitHub/TensorOperations.jl/pull/293), [#295](https://github.com/QuantumKitHub/TensorOperations.jl/pull/295), [#296](https://github.com/QuantumKitHub/TensorOperations.jl/pull/296)).

### Fixed

- Plan leak in the cuTENSOR implementation of `tensortrace!`, which retained the reduction workspace until finalization ([#294](https://github.com/QuantumKitHub/TensorOperations.jl/pull/294)).

### Performance

- `tensoradd!` bypasses `PermutedDimsArray` for `Diagonal` arguments ([#297](https://github.com/QuantumKitHub/TensorOperations.jl/pull/297)).
