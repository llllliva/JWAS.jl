# Single-Trait BayesC Inner/Outer PIP Design

## Goal

Extend `inner_outer_diagnostics=true` to single-trait dense BayesC fast-block
models, including Annotated BayesC. Preserve the existing outer-chain
`Model_Frequency` and add a PIP computed from every post-burn-in inner marker
update.

## Semantics

For marker `j`, record its effect and binary inclusion indicator immediately
after each update performed inside a fast-block repetition. The reported
inner/outer PIP is

```text
InnerOuter_Model_Frequency[j] = included inner draws[j] / inner draws[j].
```

The diagnostic includes all post-burn-in inner updates and does not depend on
`output_samples_frequency`. The existing `Model_Frequency` remains the mean of
the marker state at retained outer iterations, so both estimands remain
available for comparison.

The final, possibly shorter block has fewer repetitions under the legacy
BayesC fast-block schedule. Draw counts are therefore stored per marker and
reported in `InnerOuter_Samples` rather than assuming a common denominator.

## Implementation

- Reuse the existing compact per-marker inner/outer accumulator currently used
  by BayesR, with two BayesC states: class 1 is excluded and class 2 is
  included.
- Pass the accumulator, outer iteration, and burn-in into sequential and
  independent BayesC block samplers.
- Record each marker immediately after its binary BayesC update.
- Add `InnerOuter_Samples`, `InnerOuter_Estimate`, `InnerOuter_SD`, and
  `InnerOuter_Model_Frequency` to the BayesC marker-effects table.
- Keep BayesR class-specific diagnostic columns unchanged.

## Eligibility and compatibility

`inner_outer_diagnostics=true` requires:

- `fast_blocks != false`;
- single-trait marker models;
- dense genotype storage; and
- BayesC or BayesR.

BayesA, BayesB, GBLUP, streaming genotypes, multi-trait models, and random
regression remain unsupported. Runs with diagnostics disabled retain their
existing dispatch and output.

## Resource impact

BayesC diagnostics require two `UInt64` state counts and three per-marker
vectors for effect sums, squared-effect sums, and draw counts: approximately
40 bytes per marker, excluding small object overhead. No inner-draw history is
stored.

## Verification

- Unit-test direct BayesC block recording and per-marker draw counts.
- Run unannotated and Annotated BayesC through `runMCMC` with diagnostics.
- Assert the new PIP is finite, bounded in `[0,1]`, and based on more draws
  than the retained outer-only summary.
- Assert invalid non-fast-block, multi-trait, and streaming configurations are
  rejected.
- Confirm BayesR diagnostic tests remain unchanged and passing.
