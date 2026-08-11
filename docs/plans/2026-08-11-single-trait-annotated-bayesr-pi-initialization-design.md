# Single-Trait Annotated BayesR Pi Initialization Design

## Goal

Initialize the three nested probit models used by single-trait annotated BayesR
from the requested four-class starting `Pi` while leaving every MCMC update
unchanged.

## Problem

The annotation container currently repeats the requested `Pi` into `snp_pi`, but
its coefficients and latent means start at zero. The nested probit model therefore
implies equal conditional probabilities of `0.5`, which is inconsistent with the
marker prior stored in `snp_pi`.

## Design

When `build_model` confirms that an annotated BayesR genotype term is
single-trait:

1. Resolve the default BayesR `Pi` when the user supplied the `0.0` sentinel.
2. Validate that the four probabilities are finite, nonnegative, sum to one, and
   define all three nondegenerate nested splits.
3. Compute
   `p1 = pi2 + pi3 + pi4`,
   `p2 = (pi3 + pi4) / p1`, and
   `p3 = pi4 / (pi3 + pi4)`.
4. Zero every annotation coefficient, then set only the three intercepts to
   `quantile(Normal(), pk)`.
5. Recompute `mu` and rebuild `snp_pi` through the production nested-prior
   reconstruction helper.
6. Assert that the mean rebuilt class probabilities reproduce the starting `Pi`
   within `1e-8`, and report the initialization diagnostics.
7. Preserve a copy of the resolved starting `Pi` in `annotation_start_pi`.

The initialization runs during model finalization, once before MCMC. The Step 1,
Step 2, and Step 3 intercepts remain free in the existing Gibbs updates.

## Scope

This change applies only to single-trait annotated BayesR. It does not alter
BayesR gamma values, mixture proportions, annotations, variance settings,
multi-trait initialization, marker sampling, or annotation MCMC transitions.

## Tests

- Verify the example `Pi = [0.995, 0.004, 0.0009, 0.0001]` produces the expected
  three intercepts, zero slopes, consistent `mu`, and reconstructed `snp_pi`.
- Verify the production `get_genotypes` plus `build_model` path retains the
  starting `Pi` and reaches the same consistent state.
- Verify invalid length, negative entries, invalid sums, and degenerate nested
  splits are rejected.
- Retain the existing dense and fast-block annotated BayesR smoke tests to ensure
  the sampling path remains operational.
