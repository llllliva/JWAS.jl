# Annotation enrichment MCMC files design

## Goal

When annotation enrichment is active, save the current annotation heritability
and enrichment draws in addition to the existing posterior summary table.

## Files and layout

Create one pair per eligible annotated genotype term:

- `MCMC_samples_annotation_h2_<term>.txt`
- `MCMC_samples_annotation_enrichment_<term>.txt`

The first row contains annotation labels. Every subsequent row is one retained
MCMC draw, governed by the existing burn-in and `output_samples_frequency`.
For multiple traits, columns are ordered by trait and then annotation and are
named `<trait>_<annotation>`.

## Sampling semantics

Store the current per-draw heritability and enrichment vectors in the existing
online enrichment accumulator. This avoids recalculating the overlap-aware
probit attribution during file output. Set a trait row to `NaN` before each
attempt so a numerically invalid draw cannot repeat values from the previous
iteration.

The existing end-of-chain posterior summary remains unchanged and continues to
use all valid post-burn-in iterations, including iterations not retained in the
MCMC files.

If a retained iteration has zero or otherwise invalid total marker variance,
write `NaN` across that trait's annotation columns. This preserves iteration
alignment and avoids carrying forward values from an earlier draw.

## Tests

- Confirm both files are created by default for binary annotated BayesC.
- Confirm headers equal the reported annotation labels.
- Confirm row count follows burn-in and `output_samples_frequency`.
- Confirm each sampled heritability/enrichment row satisfies the conserved
  overlap-partition identities.
- Confirm ineligible/unannotated terms do not create these files.
