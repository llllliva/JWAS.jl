# Annotated BayesC overlap-enrichment benchmark

## Configuration

- Julia 1.10.2, one thread, Apple M2 Pro
- 100,000 markers
- 20 binary annotations
- 160,404 active annotation memberships
- deliberately included overlapping and unannotated markers
- dense single-trait annotated BayesC production accumulation path
- warm measurement after compilation

Run with:

```sh
julia --project=. --startup-file=no benchmarks/annotated_bayesc_overlap_enrichment.jl
```

## Result

- One overlap-aware enrichment update: 0.0106 seconds
- Allocations per update: 801,520 bytes
- Reported groups: 21, including `Unannotated`
- Sum of posterior heritability proportions: 1.0000000000000002
- SNP-fraction-weighted sum of enrichments: 1.0000000000000002

The per-update allocation is dominated by the required `Float64` vector of
per-marker additive variance (about 800 KB for 100,000 markers). The algorithm
does not construct or retain a marker-by-annotation allocation matrix.

Linear extrapolation to 1,385,033 markers gives about 11.1 MB of transient
per-marker variance storage. Runtime depends on annotation count, membership
density, CPU, and coefficient values, so production timing should be measured
on the target node rather than inferred solely from this small benchmark.
