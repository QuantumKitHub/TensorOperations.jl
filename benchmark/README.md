# TensorOperationsBenchmarks

Extensible benchmark suite for TensorOperations.jl. Downstream packages (e.g. a
symmetric/block-sparse tensor package) can plug in their own tensor type and run the same
standardized shapes.

## Running

```julia
julia --project=. -e '
    using TensorOperationsBenchmarks
    using TensorOperations: StridedNative, StridedBLAS
    providers = [ArrayProvider{Float64}(; backend=StridedNative()),
                 ArrayProvider{Float64}(; backend=StridedBLAS())]
    suite = build_suite(providers)
    results = run(suite)
    rows = resultstable(results)   # includes gflops/gbps (measured) and intensity (flops/bytes, static)
'
```

Or for commit-to-commit regression comparison via PkgBenchmark. Both scripts are `@main` apps
(Julia 1.11+), so `--help` works and `ARGS` are parsed the normal way:

```
julia --project=. scripts/run_benchmarks.jl --threads 1 4 --blas-threads 1 4
julia --project=. scripts/show_benchmarks.jl results_t4_blas4_strided.json   # requires CairoMakie
```

## Categories

- `:contract` -- generic pairwise contractions: a synthetic parametric shape family (tagged
  `synthetic`), 24 real quantum-chemistry contractions (CCSD, CCSD(T), AO2MO, INTENSLI) from the
  [TCCG benchmark](https://github.com/HPAC/tccg) (tagged `tccg`), and a batch of independent
  small contractions (tagged `batched`, tenferro-rs's `bij,bjk->bik` motif -- no fused
  batched-GEMM primitive exists here, so it's real per-call dispatch overhead, not one bigger
  call). Each synthetic shape also comes in up to 5 label-order layouts: `gemm_ready` is directly
  reshapeable to a BLAS call; `a_permuted`/`b_permuted`/`both_permuted` interleave open and
  contracted labels so no reshape or transpose flag suffices; `contract_scrambled` keeps the
  contracted indices contiguous in both operands but in a *different relative order* between
  them (TAPP's `D[a,d,e] = A[a,b,c]*B[c,d,e,b]`), which still isn't reshapeable despite looking
  like it should be. Every case (any category) is auto-tagged `blas` when `isblasequivalent`
  holds -- exactly `gemm_ready` among the above, but computed structurally, so e.g. TCCG
  equations that happen to be GEMM-ready get it too.
- `:permute` -- permutation-only (`tensorcopy!`) cost.
- `:trace` -- partial and full traces.
- `:mixed_precision` -- differing input/output element types (e.g. `Float32 x Float32 ->
  Float64`, mixed real/complex).
- `:network` -- multi-tensor-network motifs, tagged by `topic`: `mps` (MPS/MPO DMRG
  effective-Hamiltonian, 1-site and 2-site "theta", swept over bond `D`), `ctmrg` (CTMRG
  corner-growth step for 2D PEPS, swept over environment bond `chi`), `trg` (TRG plaquette
  contraction, swept over bond `chi`).

Not yet implemented, but addable without a redesign: MERA, contraction-order/path-finding timing.

## Adding a category

New file under `src/categories/`, define `mysizes -> Vector{BenchmarkCase}` building
`ContractSpec`/`TraceSpec`/`AddSpec`/`NetworkSpec` values, `include` it, call
`register_category!(:mycategory, mygenerator)`. Nothing else changes. Filtering uses
`BenchmarkGroup`'s native tags (`suite[@tagged "..."]`, see `BenchmarkCase`'s docstring for how
tags are derived) -- no bespoke filter API to learn.

## Plugging in a downstream tensor type

```julia
struct MyProvider <: AbstractProvider end
TensorOperations.scalartype(::MyProvider) = Float64
TensorOperationsBenchmarks.randtensor(::MyProvider, labels, dims, T) = # build a random tensor
# optional: backend(p), allocator(p), label(p), supports(p, category), rng(p)
```

Then `build_suite([MyProvider(), ArrayProvider{Float64}()])` compares directly against
TensorOperations.jl's own backends. `ArrayProvider` shows the recommended `randtensor` pattern:
allocate via `TensorOperations.tensoralloc` (goes through `allocator`) and fill from a stored,
stateful `rng` field seeded once at construction, so runs are reproducible but tensors within a
run still differ.

## Threading and precision

Thread config is not a `build_suite` axis -- setting it per-case would count the switch itself
as part of the timing. Instead `set_threads!` is called once, at the process level, before a
suite is built (`benchmarks.jl` reads `TOB_BLAS_THREADS`/`TOB_STRIDED_THREADS`, set by
`run_benchmarks.jl --blas-threads/--strided-threads`). `with_threads(f, cfg)` is available for
ad hoc comparisons, wrapping a whole `run(suite)` call and restoring afterwards.

Mixed precision: set `TA`/`TB`/`TC` explicitly on a spec (see `mixed_precision.jl`); other
categories leave them `nothing` (provider's default `scalartype`).
