# Default annotation enrichment output implementation

## Changes

- Changed the documented and executable `runMCMC` default for
  `output_annotation_enrichment` from `false` to `true`.
- Binary annotated genotype terms now automatically accumulate, return, and
  write their annotation heritability-enrichment summaries without requiring
  the keyword.
- Models without annotated genotype terms remain a no-op for enrichment.
- Continuous annotation covariates remain valid for the probit sampler, but are
  skipped by automatic enrichment output because enrichment requires discrete
  binary 0/1 groups.
- Existing random-regression and single-step restrictions are applied only when
  an eligible binary annotated term is present.
- Explicit `output_annotation_enrichment=false` still disables the calculation.

## Validation

- The focused enrichment MCMC tests omit the keyword and verify that the output
  table and text file are produced: 45/45 tests passed.
- The annotated BayesC API suite, including continuous annotations, passed
  106/106.
- The complete JWAS unit suite passed 1,131/1,131 on Julia 1.10.2.
- The Documenter manual build completed successfully; deployment was skipped as
  expected because the copied directory is not a configured CI checkout.
