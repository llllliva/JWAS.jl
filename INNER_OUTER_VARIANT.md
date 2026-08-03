# BayesR inner-plus-outer diagnostic variant

This sibling package is based on `JWAS_BayesR_per_class_Pi_out` at commit
`1e78f62c`. The original package is unchanged.

## Enable the diagnostic

Pass this package to the submission runner:

```text
--jwas_path /path/to/JWAS_BayesR_inner_outer_info
```

and add the following `runMCMC` keyword:

```julia
inner_outer_diagnostics=true
```

The option currently requires single-trait, dense-genotype BayesR with
`fast_blocks != false`.

The four class-variance multipliers can be changed per genotype category with
`get_genotypes(...; BayesR_gamma=[0.0, 0.001, 0.01, 0.1])`. The null class must
remain zero and the three non-null values must be positive and strictly
increasing. Custom gamma vectors are currently restricted to single-trait
BayesR, matching the inner--outer diagnostic restriction.

For a numeric/default fast-block size, JWAS converts the requested
`chain_length` to an outer-chain length using
`floor(chain_length / block_size)`. It does not convert `burnin`. Specify
`burnin` in outer-iteration units and keep it smaller than the reported
outer-chain length. This variant raises an explicit error otherwise.

## Output

The existing marker-effect columns remain outer-chain summaries:

```text
Estimate
SD
Model_Frequency
Mean_Delta
Class1_PIP ... Class4_PIP
```

When the diagnostic is enabled, `marker_effects_<name>.txt` also contains:

```text
InnerOuter_Samples
InnerOuter_Estimate
InnerOuter_SD
InnerOuter_Model_Frequency
InnerOuter_Mean_Delta
InnerOuter_Class1_PIP ... InnerOuter_Class4_PIP
```

For every marker, an inner/outer sample is recorded after each completed local
block sweep after burn-in. The final local sweep is also the state visible at
the outer boundary, so it is counted once rather than counted again as a
separate outer observation.

Blocks may have different sizes, especially the final block. Consequently,
`InnerOuter_Samples` is stored per marker and must be used as the denominator.

## Interpretation

The extra columns improve the resolution of marker-level effect and class-PIP
summaries. They are conditional local-block draws: markers outside the current
block remain fixed during those repetitions. They are therefore useful
diagnostics and provide a BayesR3-like inner-sweep summary, but they must not be
reported as additional independent whole-genome posterior samples.

Pi, shared marker variance, residual variance, genomic variance, and other
global quantities are still updated and summarized at the outer-chain level.
Updating those global parameters inside a local block would change the Markov
chain rather than merely calculate additional information.
