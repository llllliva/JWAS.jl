# Single-step Float32 memory benchmark

## Scope

The benchmark exercises the same low-memory imputation calculation used by
`SSBR.jl` in fresh Julia 1.10.2 processes.  It compares the historical
whole-`Mg`/`Float64` destination implementation with the new chunked
`Float32` destination implementation.

Synthetic dimensions:

- 4,000 genotyped animals;
- 500 non-genotyped animals;
- 4,500 pedigree/phenotype rows;
- 500 output animals;
- 10,000 markers;
- 1,000 markers per imputation chunk.

Run with:

```bash
julia --project=. --startup-file=no \
  benchmarks/benchmark_single_step_imputation_memory.jl legacy
julia --project=. --startup-file=no \
  benchmarks/benchmark_single_step_imputation_memory.jl optimized
```

`Sys.maxrss()` was captured inside each fresh process.

## Results

| Mode | Checksum | Maximum RSS |
|---|---:|---:|
| Historical | `2.3753834e7` | 1.782 GiB |
| Chunked Float32 | `2.3753834e7` | 1.148 GiB |

The optimized path reduced peak RSS by 0.634 GiB, or 35.6%, in this
moderate-scale benchmark.  The identical checksums complement the unit test's
elementwise comparison with the historical formula.

## Production interpretation

The saving grows with the number of genotyped animals and markers because the
new implementation never creates the complete `Zgeno*genotypes` matrix.  In
the historical code, multiplying the `Float64` sparse incidence matrix by a
`Float32` dense genotype matrix also promotes that complete reordered matrix
to `Float64`.

For the 55,255 by 1,385,033 production observed-genotype matrix, avoiding that
promoted reordered matrix removes roughly 570 GiB.  Storing the 70,171 aligned
phenotype/output rows directly as `Float32` instead of first retaining
`Float64` versions removes roughly another 362 GiB from steady storage and
avoids full-size conversion copies.  Actual job RSS will also contain Julia,
sparse factorization, sampler, annotation, and output state, so a full-scale
one-iteration pilot remains necessary before lowering the SLURM request.
