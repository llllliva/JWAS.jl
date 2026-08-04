# BayesR Runtime Options Implementation

## Implemented Changes

- Added package-level resolvers for `BAYESR_BLOCK_STARTS` and
  `BAYESR_INNER_ITERATIONS`.
- Integrated resolution at the beginning of `runMCMC` for BayesR analyses.
- Preserved explicit keyword precedence and all behavior when the environment
  variables are absent.
- Kept automatic inner repetitions as the existing per-block marker-count
  calculation, with one repetition during burn-in.
- Added input validation before output-directory creation.
- Extended focused tests to cover resolver validation, environment fallback,
  explicit precedence, prevention of unintended outer-chain rescaling, fixed
  inner repetitions, and automatic unequal-block repetitions.

## Farm Driver Compatibility

Older drivers may continue passing `fast_blocks=true` and
`BayesR_inner_iterations=false`. The package now consumes the environment
values exported by the submission wrapper, so those drivers do not need to
duplicate parsing logic.

## Verification

- Focused BayesR runtime/diagnostic tests: 28 passed.
- Full JWAS unit suite: 1,016 passed.
- Documentation build was attempted but the local docs environment does not
  have `Documenter` installed.
