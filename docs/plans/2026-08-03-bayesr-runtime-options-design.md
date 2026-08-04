# BayesR Runtime Options Design

## Problem

Farm submission wrappers export chromosome block starts and an optional fixed
BayesR inner-iteration count, but existing analysis drivers may still call
`runMCMC` with `fast_blocks=true` and `BayesR_inner_iterations=false`.
Consequently, JWAS applies the automatic square-root block size, rescales the
outer chain, and can reduce a short requested chain to zero.

## Design

JWAS will resolve the two BayesR runtime options at the beginning of `runMCMC`,
before output directories are created:

- When `fast_blocks === true` and `BAYESR_BLOCK_STARTS` is present, parse the
  comma-separated, 1-based block starts and use that explicit vector.
- Explicit numeric or vector `fast_blocks` arguments continue to take
  precedence over the environment.
- When `BayesR_inner_iterations === false`, consult
  `BAYESR_INNER_ITERATIONS`.
  - A missing, empty, `false`, or `auto` value retains automatic calculation.
  - A positive integer sets the same fixed inner repetition count for every
    block.
- An explicit positive `BayesR_inner_iterations` argument takes precedence
  over the environment.

Automatic inner calculation remains the existing BayesR behavior: burn-in uses
one repetition and each post-burn-in block uses its own marker count as the
number of repetitions.

## Validation

Environment block starts must begin at 1, contain positive integers, and be
strictly increasing. The existing marker-count validation remains in
`validate_fast_block_starts`. Inner iterations must resolve to `false` or a
positive integer. Invalid environment values fail before analysis output is
created.

## Compatibility

Runs without these environment variables are unchanged. Direct keyword
arguments continue to work. The environment fallback is intentionally limited
to `fast_blocks=true` and `BayesR_inner_iterations=false`, which are the values
passed by older Farm drivers.

## Tests

Tests will cover missing/default values, explicit keyword precedence, valid and
invalid environment values, explicit chromosome starts preventing chain
rescaling, fixed environment inner iterations, and automatic block-size inner
iterations when no fixed count is supplied.
