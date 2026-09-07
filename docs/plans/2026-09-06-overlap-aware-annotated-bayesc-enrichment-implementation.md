# Overlap-aware annotated BayesC enrichment implementation

## Changes

- Replaced full repeated counting of overlapping SNP variance for single-trait
  annotated BayesC with a conserved within-SNP partition.
- Each active annotation is weighted by the absolute leave-one-out change it
  produces in the current probit inclusion probability. Weights are normalized
  within SNP; a zero-score SNP uses equal weights.
- Added an `Unannotated` reporting group when any post-QC SNP has no active
  annotation. If the input already contains an explicit `Others` annotation,
  no synthetic group is added for those SNPs.
- Changed the enrichment baseline from overlapping raw membership fractions to
  fixed equal fractional membership. `SNP_Count` remains the raw membership
  count, while `SNP_Fraction` is the conserved effective fraction used in the
  enrichment denominator.
- Added `Allocation_Method` to the output table. The new path reports
  `probit_leave_one_out_partition`; unchanged BayesR/multi-trait paths report
  `marginal_full_counting`.
- Added method documentation, conservation tests, and a production-path
  microbenchmark.

The sampler itself is unchanged. Annotation coefficients, marker-specific pi,
marker effects, and total marker heritability are sampled exactly as before;
only posterior attribution to enrichment groups changes.

## Per-draw invariants

For every valid single-trait annotated BayesC draw:

- annotation heritability proportions sum to one;
- annotation heritabilities sum to total marker heritability;
- the SNP-fraction-weighted enrichment sum equals one;
- every SNP's additive variance enters the group totals exactly once.

## Validation

- Focused annotation enrichment tests: 45/45 passed before the full run.
- Full JWAS unit suite: 1,131/1,131 passed on Julia 1.10.2.
- Documenter manual build completed successfully; deployment was skipped outside
  CI because this copied folder has no Git remote metadata.

## Performance

The warm benchmark at 100,000 SNPs, 20 annotations, and 160,404 memberships took
0.0106 seconds and allocated 801,520 bytes per enrichment update. It reported
21 groups including `Unannotated`, with both conservation sums equal to
1.0000000000000002. Linear extrapolation gives about 11.1 MB for the transient
per-marker variance vector at 1,385,033 SNPs. No SNP-by-annotation allocation
matrix is materialized.

See `benchmarks/reports/2026-09-06-annotated-bayesc-overlap-enrichment.md` for
the benchmark configuration and interpretation.
