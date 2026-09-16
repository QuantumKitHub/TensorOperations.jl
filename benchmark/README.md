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
    rows = resultstable(results)
'
```

Or for commit-to-commit regression comparison via PkgBenchmark. Both scripts are `@main` apps
(Julia 1.11+), so `--help` works and `ARGS` are parsed the normal way:

```
julia --project=. scripts/run_benchmarks.jl --threads 1 4 --blas-threads 1 4
julia --project=. scripts/show_benchmarks.jl results_t4_blas4_strided.json   # requires CairoMakie
```

## Categories (v1)

- `:pairwise` -- generic pairwise contractions: a synthetic parametric shape family (tagged
  `synthetic`) plus 24 real quantum-chemistry contractions (CCSD, CCSD(T), AO2MO, INTENSLI) from
  the [TCCG benchmark](https://github.com/HPAC/tccg) (tagged `tccg`). Each synthetic shape also
  comes in up to 4 label-order layouts (tagged `gemm_ready`/`a_permuted`/`b_permuted`/
  `both_permuted`): `gemm_ready` is directly reshapeable to a BLAS call, the others interleave
  open/contracted labels so no reshape or transpose flag suffices -- a real permutation is
  required, which is what actually separates `StridedNative` from `StridedBLAS`.
- `:permute` -- permutation-only (`tensorcopy!`) cost.
- `:trace` -- partial and full traces.
- `:mixed_precision` -- differing input/output element types (e.g. `Float32 x Float32 ->
  Float64`, mixed real/complex).
- `:mps` -- MPS/MPO DMRG effective-Hamiltonian motif (1-site and 2-site "theta"), swept over
  bond dimension `D`.
- `:ctmrg` -- CTMRG corner-growth step (2D PEPS boundary-MPS), swept over environment bond `chi`.
- `:trg` -- TRG plaquette contraction (4-ring of rank-3 tensors), swept over bond `chi`.

Not yet implemented, but addable without a redesign: MERA, contraction-order/path-finding timing.

## Adding a category

New file under `src/categories/`, define `mysizes -> Vector{BenchmarkCase}` building
`ContractSpec`/`TraceSpec`/`AddSpec`/`NetworkSpec` values, `include` it, call
`register_category!(:mycategory, mygenerator)`. Nothing else changes.

## `BenchmarkCase` vs. plain `BenchmarkTools`

A `BenchmarkTools.Benchmark`/`Trial` only knows how to run a closure and record timings -- it
carries no metadata about *why* that closure exists. `BenchmarkCase` is our own struct that
keeps the `AbstractCaseSpec` (needed for the `flops`/`bytes` cost model) and `params`
(the sweep values that produced it) alongside each case, so `resultstable` can join timings
back against cost figures after the fact. `build_suite` consumes a `Vector{BenchmarkCase}` and
produces an ordinary `BenchmarkGroup`; nothing downstream of that ever sees `BenchmarkCase`
again.

Filtering uses `BenchmarkGroup`'s native tags, not a bespoke mechanism: each case is wrapped as
`BenchmarkGroup(casetags(case), "benchmark" => ...)`, where `casetags` is the category name plus
every `Symbol`-valued `params` entry (`source`, `kind`, `variant`, `layout`, ...). Filter the
*built* suite with `@tagged` (re-exported from BenchmarkTools) before running it:

```julia
suite = build_suite(providers)
run(suite[@tagged "tccg"])                    # only the TCCG-sourced pairwise cases
run(suite[@tagged "pairwise" && "both_permuted"])  # boolean tag expressions work
```

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
