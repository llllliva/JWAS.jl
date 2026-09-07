# Shared Dense Multiclass and Row-Oriented Annotated Pi Output Design

## Goals

1. Allow overlapping BayesC genotype classes to share one already-preprocessed
   dense genotype matrix rather than materializing one matrix per class.
2. Write single-trait Annotated BayesC marker-specific pi samples with one
   retained MCMC iteration per row and one marker per column.

## Shared dense multiclass API

Add `get_shared_genotype_classes(source, memberships; ...)`, where `source` is
a dense `Genotypes` object and `memberships` is a marker-by-class binary matrix
in the source object's post-QC marker order.

The helper will:

- validate that the source is dense marker data, not a GRM;
- optionally add an `Others` class for markers having no supplied membership;
- create a zero-copy column view of `source.genotypes` for each class;
- copy only small per-class metadata and create independent BayesC priors and
  MCMC state;
- return one `Genotypes` object per class for use as ordinary JWAS model terms.

The source object must be created and preprocessed once before class views are
created. Class objects must not repeat centering, imputation, or marker QC.
Overlapping classes share genotype columns but retain distinct marker effects,
inclusion indicators, pi, and marker-effect variances, preserving multiclass
BayesC behavior.

Dense matrix fields and Gibbs-matrix setup will accept `AbstractMatrix` so
column views can flow through the existing samplers. Exact genotype/phenotype
ID order preserves sharing through analysis setup. When row alignment is
required, shared class matrices will be aligned with row views instead of
dense matrix multiplication; ordinary owned matrices retain existing behavior.

This feature changes storage only. It does not resolve the statistical
non-identifiability of separate effects for a physical SNP duplicated across
overlapping classes.

## Annotated BayesC pi samples

For single-trait Annotated BayesC, `Mi.π` is a marker-specific vector. The
current generic `writedlm` call writes that vector vertically and inserts a
blank separator, causing each iteration to span many lines.

Change only this case to:

- write marker IDs as the pi sample-file header;
- transpose the marker-specific pi vector at output time;
- omit blank separator lines;
- preserve current scalar, BayesR, unannotated, and multi-trait formats.

The resulting file shape is `(retained samples + 1) × nMarkers`, including the
header row.

## Tests

- Verify overlapping classes alias the same parent genotype matrix and contain
  the expected marker columns without allocating class matrices.
- Verify class effects and priors remain independent.
- Run a small ordinary multiclass BayesC analysis using shared views.
- Verify single-trait Annotated BayesC pi files have marker headers and exactly
  one row per retained iteration.
- Verify ordinary BayesC scalar pi output remains unchanged.
