# Shared Dense Multiclass Memory Smoke Benchmark

## Scope

The production JWAS `get_genotypes` -> `get_shared_genotype_classes` ->
`build_model` -> `runMCMC` path was exercised with overlapping BayesC classes.
This is a local smoke benchmark, not a performance estimate for the full cattle
dataset.

Command:

```text
julia --project=. --startup-file=no benchmarks/shared_dense_multiclass_memory.jl
```

Environment: Julia 1.10.2, macOS arm64, one Julia thread.

## Result

Synthetic dimensions were 100 observations by 1,000 physical markers, with
1,668 total marker-class memberships across three overlapping classes.

| Measurement | Result |
|---|---:|
| One Float32 genotype matrix | 400,000 bytes |
| Logical storage if class matrices were copied | 667,200 bytes |
| Warm shared-class construction allocation | 56,192 bytes |
| All class matrices alias the base | true |
| Warm class-heritability calculation allocation | 7,040 bytes |
| Two-iteration production MCMC smoke time | 7.02 seconds |

The construction allocation includes marker-index and small metadata arrays;
it does not scale as observations times memberships. All class genotype
matrices remained views into the one base matrix, including after phenotype row
reordering in the regression test.

The class-heritability calculation uses an observation-by-class matrix of
genetic values rather than copying genotype columns. Its measured allocation
was 7,040 bytes for 100 observations and three classes; the expected leading
temporary storage is approximately `8 * nobs * (nclasses + 1)` bytes.

For the target annotation table, the measured membership ratio is about
2.5457. If one complete dense genotype matrix is 300 GB, copied class matrices
would represent about 764 GB of genotype values alone. Shared dense storage
keeps that component at 300 GB. MCMC coefficient arrays, marker indices, view
objects, block cross-products, and Julia runtime allocations remain additional
memory and require measurement with the production job's `MaxRSS`.
