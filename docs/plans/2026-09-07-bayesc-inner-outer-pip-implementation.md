# BayesC Inner/Outer PIP Implementation

Date: 2026-09-07

## Implemented behavior

`inner_outer_diagnostics=true` now supports single-trait dense BayesC,
including annotated BayesC, when fast blocks are enabled. Each post-burn-in
inner marker update is accumulated online as excluded (`delta=0`) or included
(`delta=1`). The implementation reuses the compact BayesR inner/outer
accumulator with two BayesC states; it does not save an inner MCMC history.

Both sequential and independent fast-block update paths record diagnostics.
Each marker keeps its own draw denominator, which correctly handles a final
block that is smaller than the requested block size.

The marker-effects result and `marker_effects_<term>.txt` now contain these
common columns for BayesC and BayesR when diagnostics are enabled:

- `InnerOuter_Samples`
- `InnerOuter_Estimate`
- `InnerOuter_SD`
- `InnerOuter_Model_Frequency`

The existing outer-sweep `Model_Frequency` remains unchanged. BayesR retains
its additional mean-delta and class-specific inner PIP columns.

## Validation

- Added an end-to-end annotated single-trait BayesC test covering output
  columns, marker-specific sample counts, bounds, finite summaries, output-file
  creation, and option validation.
- Added direct coverage of independent-block BayesC accumulation.
- Existing BayesR inner/outer diagnostic tests continue to pass.
- Full unit suite: 1,175 of 1,175 tests passed in 2m51s.
- Documentation build completed successfully. Documenter emitted only its
  expected local-build warnings about remote-branch detection and deployment.

## Performance

The accumulator requires 40 bytes per marker, independent of chain length. A
production-path benchmark with 5,000 markers, 300 outer iterations, and three
paired seeds measured 14.3% median runtime overhead. See
`benchmarks/reports/2026-09-07-bayesc-inner-outer-diagnostics.md` for the full
configuration and results.
