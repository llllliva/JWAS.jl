# Default annotation enrichment output design

## Goal

Make `output_annotation_enrichment` default to `true` so annotated BayesC runs
automatically accumulate and write their enrichment summary.

## Compatibility rule

A global `true` default must remain safe for ordinary models without annotated
genotype terms. Such models silently perform no annotation-enrichment work and
create no enrichment file. Restrictions such as random-regression and
single-step incompatibility are checked only when an annotated term is actually
present and eligible. Continuous annotation covariates remain valid for the
probit sampler but cannot define discrete enrichment groups, so the automatic
enrichment output skips them. Users can still set
`output_annotation_enrichment=false` to disable the calculation for annotated
models.

## Validation

- Verify an annotated BayesC run without the keyword returns and writes the
  enrichment table.
- Verify an ordinary unannotated run without the keyword still completes and
  does not return an enrichment table.
- Run the full unit suite and documentation build.
