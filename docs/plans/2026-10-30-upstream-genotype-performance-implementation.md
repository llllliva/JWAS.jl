# Upstream genotype performance implementation and compatibility

Base: latest fetched `reworkhow/JWAS.jl` master
`038c6c8a497ec6e17e957657a5c13ae698377132` (JWAS 2.4.0), 2026-10-30.
Branch: `performance/upstream-genotype-views-residual`.

Only storage adoption, contiguous row/block views and residual precision
optimizations were ported. Current upstream annotation and sampling code was
retained. BayesR and independent blocks are existing upstream features; both
now receive the same precision-matched residual multipliers. Shared mode
supports upstream's single-trait dense BayesC and BayesR. Import ownership for
files/DataFrames/streaming remains unchanged. No dependency or version bump.

Upstream numeric fast blocks divide chain length but leave burn-in in outer
units. The fork used a different convention. The port preserves upstream's
behavior, and the production test fixtures were adapted to it.

## Intentional API incompatibilities of the requested defaults

| Default | Consequence | Explicit compatibility option |
| --- | --- | --- |
| `copy_genotypes=false` | Matching-precision caller matrices can be mutated by preprocessing. | `copy_genotypes=true` preserves ownership. |
| `copy_genotypes=false` | Mismatched-precision or integer matrices cannot be adopted. | `copy_genotypes=true` permits conversion. |
| `shared_genotypes=true` | Requires one dense single-trait BayesC/BayesR term, matching precision, unit weights, and ordered contiguous unique row ranges. | `shared_genotypes=false` retains old alignment for other models/layouts. |

Streaming, multi-trait, single-step, random-regression, GBLUP/BayesA/BayesB/
BayesL/RR-BLUP and multiple-genotype-term analyses must explicitly disable
sharing. Repeated/permuted/noncontiguous rows also need that opt-out.
Conventional no-marker models skip sharing validation and continue working.
These defaults should be called out as a breaking API change in an upstream PR.

## Validation

- Untouched upstream: all 988 checks passed.
- Unmodified upstream tests against the new defaults: 779 passed, 5 failed,
  42 errored, confirming default-related incompatibilities.
- Port with explicit legacy opt-outs and new default/production/residual
  coverage: all 2352 checks passed. No existing assertions were removed.
- Documentation build passed from the port's verified source path.
- Eight direct upstream/port comparisons passed keyed numerical tolerances;
  ordinary outputs matched exactly and maximum fast-block EBV difference was
  4.046e-6. Eight copied/shared storage comparisons had identical outputs and
  42.9% lower retained genotype payload.

Tests used Julia 1.11.7 on Apple Silicon. Optional integration tests and the
full upstream CI runtime/OS matrix were not run. The synthetic comparisons
are correctness checks rather than convergence or peak-memory measurements.
See `benchmarks/reports/2026-10-30-upstream-genotype-performance/README.md`
for methods, metrics and reproduction instructions.

The old fork-based review branch is preserved. Neither master branch is changed,
and no upstream pull request is submitted during this review.
