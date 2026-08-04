# BayesR gamma environment resolution design

## Problem

`runMCMC` accepts a `BayesR_gamma` keyword, but an exported
`BAYESR_GAMMA` environment value is currently ignored when the keyword is
omitted. This allows command-line drivers to advertise a custom gamma while
JWAS silently uses its compiled default.

## Design

Add `resolve_bayesr_gamma(gamma=nothing; env=ENV)` at the JWAS API boundary.
Resolution order is:

1. A non-`nothing` `BayesR_gamma` keyword, validated with the existing rules.
2. A non-empty `BAYESR_GAMMA` environment value parsed as four
   comma-separated floating-point values and validated with the same rules.
3. A copy of the existing `BAYESR_GAMMA` package constant.

Change the `runMCMC` keyword default from `BAYESR_GAMMA` to `nothing`, allowing
JWAS to distinguish an omitted keyword from an explicit value. Explicit
keywords always override the environment. Missing or whitespace-only
environment values preserve existing behavior.

## Validation and compatibility

- Retain the existing four-class, null-first, positive, finite, strictly
  increasing validation.
- Invalid environment syntax or values fail before MCMC starts.
- No environment variable means the numerical default remains
  `[0.0, 0.01, 0.1, 1.0]`.
- Add unit coverage for fallback, environment parsing, explicit precedence,
  whitespace, and malformed values.
- Exercise the environment path through a small `runMCMC` integration test and
  assert the resolved value in `MCMCinfo` and class-variance output.
