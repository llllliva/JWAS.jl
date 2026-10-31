# BayesC-only memory implementation

Base: reworkhow master 038c6c8a497ec6e17e957657a5c13ae698377132 (JWAS 2.4.0).
Review branch: performance/bayesc-only-shared-views.

## Included

- Ordinary and fast-block BayesC; single-trait and multi-trait samplers I/II.
- Sequential and independent blocks.
- BayesC-only matrix-adoption defaults and scoped automatic shared storage.
- Contiguous parent-backed training, prediction and block views, with parent-aware
  memory accounting. Other methods retain copying/storage defaults.
- Compatibility residual arithmetic by default; optional fast_residual_updates
  enables precision-matched in-place BLAS for BayesC blocks.
- Sequential MT block-sized alpha copies and reusable product workspaces;
  upstream product-then-add order is retained. Legacy mixed-precision calls
  fall back to the original expression to preserve promotion/evaluation.

BayesR and MTBayesR source files, prediction output code, priors, dependencies,
package version and sampling schedules are unchanged from upstream.

## Verification

The untouched legacy suite passed all 988 checks without modifying its test
cases or adding opt-outs. New focused storage/production tests passed 1089
checks; single-/multi-trait residual reconstruction passed 312 checks across
both precisions, both arithmetic modes, both block modes and three seeds.
Eight additional mixed-precision compatibility checks passed exactly.

The final combined full suite passed all 2397 checks, including the eight
additional mixed-precision checks. No original regression test assertions were
changed or disabled.

All 72 direct upstream-versus-candidate prediction cases matched exactly:
single-trait plus two-trait samplers I/II, both precisions, ordinary/sequential/
independent updates, three seeds, annotations, partial trait missingness and
held-out animals. EBV, PEV, marker posterior mean/SD and model frequencies had
zero maximum difference in the default compatibility arithmetic mode.
Faster arithmetic is deliberately opt-in and is not claimed bitwise identical.

Eight production storage comparisons (one/two traits, ordinary/sequential,
two seeds) reduced retained genotype payload from 230272 to 131584 bytes,
42.9%, when training/prediction outputs overlap. This is not total RSS.

Documentation built successfully from the candidate source; deployment skipped.
Tests used Julia 1.11.7 on Apple Silicon. Optional integration tests and the
full CI runtime/OS matrix were not run. Synthetic prediction checks do not
establish production convergence or equivalence for every dataset.

See benchmarks/reports/2026-10-31-bayesc-only/ for compact keyed evidence and
reproduction details. Existing runtime priors and formulas were preserved.
No master branch or upstream pull request is changed during this review.
