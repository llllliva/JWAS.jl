# Single-Trait Annotated BayesR Pi Initialization Implementation

## Changes

- Added `initialize_bayesr_single_trait_annotation_coefficients!` to initialize
  the three nested annotation-probit intercepts from the requested four-class
  BayesR `Pi`.
- All annotation slopes are explicitly reset to zero before the intercepts are
  assigned.
- The helper recomputes `mu`, rebuilds the per-SNP four-class priors through
  `rebuild_bayesr_nested_priors!`, and asserts that their mean reproduces the
  starting `Pi` within `1e-8`.
- Added finite-value, nonnegativity, sum-to-one, container-shape, and nested-split
  validation.
- Single-trait annotated BayesR invokes the initialization during model
  finalization, before MCMC. The resolved iteration-0 mixture is retained in
  `annotation_start_pi`, while `Mi.π` remains available for the existing MCMC
  updates.
- Added initialization diagnostics for the starting mixture, nested
  probabilities, intercepts, and reconstructed mean mixture.
- Updated the annotated BayesR manual to describe the `Pi`-consistent probit
  initialization and clarify that all coefficients remain free during MCMC.
- Left the existing Step 1--3 liability, coefficient, variance, class-prior, and
  marker-effect sampling code unchanged; all three intercepts remain free.

## Verification

Focused unit and smoke tests were run with the Julia version recorded by the
manifest:

```bash
julia +1.11.7 --project=. --startup-file=no test/unit/test_annotated_bayesr.jl
```

Result: `125` tests passed.

The tests cover the requested
`Pi = [0.995, 0.004, 0.0009, 0.0001]`, expected nested intercepts, zero slopes,
`mu` reconstruction, per-SNP prior reconstruction, input guards, and the existing
dense and fast-block MCMC paths.

The complete unit suite was also run:

```bash
julia +1.11.7 --project=. --startup-file=no test/runtests.jl
```

Result: `1058` tests passed; integration tests remained disabled by the suite's
default configuration.

Documentation verification:

```bash
julia +1.11.7 --project=docs --startup-file=no docs/make.jl
```

Result: documentation rendered successfully. Deployment was skipped because the
local build environment is not a deployment target.
