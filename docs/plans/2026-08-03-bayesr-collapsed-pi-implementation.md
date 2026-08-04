# BayesR Collapsed Pi Sampler Implementation

## Summary

JWAS now provides an opt-in collapsed mixture-proportion transition for
ordinary single-trait BayesR:

```julia
runMCMC(
    model,
    phenotypes;
    BayesR_pi_sampler=:collapsed,
)
```

The default remains `BayesR_pi_sampler=:standard`, which preserves the existing
sampled-`pi` Gibbs transition.

## Implementation

- `runMCMC` validates and stores `BayesR_pi_sampler` in `MCMCinfo`.
- Symbols and case-insensitive strings `standard` and `collapsed` are accepted.
- The dense and sequential fast-block BayesR marker kernels maintain one
  four-element class-count vector when collapsed sampling is enabled.
- Before marker `j` is updated, its old class is removed from the counts. With
  the existing `Dirichlet(1,1,1,1)` prior, its class log prior becomes
  `log(counts_without_j[k] + 1)`.
- The newly sampled class is immediately added to the global counts, including
  during repeated inner block updates.
- Collapsed runs initialize `delta` from the user-supplied starting `Pi` to
  avoid an unusable all-null initial count state. This is initialization only;
  the collapsed posterior continues to use `alpha=[1,1,1,1]`.
- JWAS continues drawing `pi ~ Dirichlet(counts + 1)` at the end of each outer
  iteration for posterior output. Collapsed marker updates do not consume this
  draw on the following sweep.
- `getMCMCinfo` and BayesR startup output identify the selected sampler.

The collapsed option is rejected for fixed `Pi`, annotated BayesR, multi-trait
BayesR, streaming storage, RRM, and independent blocks. Those configurations do
not have the single global Dirichlet state required by this implementation.

## Compatibility

The standard path retains its original initialization, positional fast-block
dispatch, random draws, and class-probability calculation. A seeded regression
test verifies that omitting the keyword and explicitly selecting `:standard`
produce identical low-level marker transitions.

## Tests

Focused collapsed-pi tests cover:

- option parsing and validation;
- class counting and current-marker exclusion weights;
- seeded legacy-default equivalence;
- dense production-path execution;
- sequential fast-block execution;
- valid pi and PIP output; and
- rejection of `estimatePi=false`.

Verification completed with Julia 1.11.7:

```text
Focused collapsed-pi tests: 20 / 20 passed
Full unit test suite:       1036 / 1036 passed
```

The matched AlphaSimR comparison and interpretation are recorded in
`benchmarks/reports/2026-08-03-bayesr-collapsed-pi-alphasimr.md`.
