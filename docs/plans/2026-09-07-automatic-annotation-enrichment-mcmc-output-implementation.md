# Automatic annotation enrichment and MCMC output implementation

## Delivered behavior

- `output_annotation_enrichment` now defaults to `true`.
- Ordinary models without annotated terms remain a no-op for enrichment.
- Annotated probit models with continuous covariates remain valid but are
  automatically skipped for enrichment because they do not define binary SNP
  groups.
- Eligible binary annotated terms retain the existing end-of-chain summary and
  now also write:
  - `MCMC_samples_annotation_h2_<term>.txt`
  - `MCMC_samples_annotation_enrichment_<term>.txt`
- Both files use annotations as columns and retained MCMC iterations as rows.
  Burn-in and `output_samples_frequency` use the same schedule as all other
  JWAS MCMC sample files.
- Multi-trait headers use `<trait>_<annotation>` ordering.
- Invalid draws with zero/nonfinite total marker variance are represented by an
  all-`NaN` trait row, preserving iteration alignment without carrying values
  forward.
- `output_annotation_enrichment=false` disables accumulation, summary output,
  and both MCMC files.

## Implementation

The enrichment accumulator stores only the current heritability and enrichment
vectors in two trait-by-group matrices. The MCMC writer emits these matrices at
the ordinary retained-sample cadence. It does not recompute overlap attribution
and does not store the chain in memory.

The new persistent state is `2 * ntraits * ngroups * sizeof(Float64)`. File size
scales with retained iterations times annotation groups, not SNP count.

## Validation

- Focused enrichment suite: 69/69 passed.
- Full JWAS unit suite: 1,155/1,155 passed on Julia 1.10.2.
- Standard and fast-block tests verify file existence, headers, retained row
  count, finite/`NaN` row semantics, heritability bounds, and conserved weighted
  enrichment for valid rows.
- Existing continuous-annotation tests confirm the automatic binary-group
  eligibility check preserves probit-model compatibility.
- Documenter manual build completed successfully; deployment was skipped
  outside CI because this copied folder has no Git remote metadata.
