# Annotation Heritability Enrichment Implementation

## Summary

JWAS now has an opt-in posterior summary for marker-additive heritability
enrichment in annotated models:

```julia
output = runMCMC(
    model,
    phenotypes;
    output_annotation_enrichment=true,
)

enrichment = output["annotation heritability enrichment geno"]
```

The default is `false`, so existing runs do not perform the extra calculation
or produce an extra result. The normal output writer also saves the table as:

```text
annotation_heritability_enrichment_geno.txt
```

## Implementation

The change adds one compact online accumulator to each annotated genotype term.
For every post-burn-in main MCMC state it computes marker contributions
`2p(1-p)alpha^2`, sums those contributions over each supplied binary annotation,
and updates posterior first and second moments. Full per-iteration traces are not
stored.

The update is located in the main sampler after marker effects and residual
variance have been updated. It is outside the marker and block kernels. Thus:

- ordinary sampling records every post-burn-in MCMC iteration;
- `fast_blocks` records every post-burn-in outer iteration, after all blocks in
  that outer sweep have been processed;
- `output_samples_frequency` does not thin the enrichment calculation.

The returned table has one row per trait and non-intercept annotation, with:

- post-QC SNP count and fraction;
- posterior mean and SD of annotation heritability;
- posterior mean and SD of the annotation's marker-heritability proportion;
- posterior mean and SD of enrichment;
- posterior probability that enrichment is greater than one;
- attempted posterior iterations and valid posterior samples.

Annotation labels are `Annotation_1`, `Annotation_2`, and so on, matching the
column order supplied to `get_genotypes`. This avoids changing the existing
annotation input API.

## Validation and Interpretation

The output requires nonempty binary `0/1` annotation columns after genotype
quality control. Overlapping groups are accepted and reported marginally, so
their proportions can sum to more than one. The internal annotation intercept
is excluded.

If an attempted state has zero total sampled marker variance, its proportion and
enrichment are undefined. That state increments `Posterior_Iterations` but is
excluded from `Posterior_Samples` and the posterior moments.

The summary is marker-additive and does not allocate LD covariance. For models
with multiple genotype terms, each table uses that term's marker variance and
the residual variance. Random-regression and single-step analyses are rejected
for this option because their additional genetic components are not represented
in this denominator.

## Verification

Verification used Julia 1.11.7, matching the repository manifest:

| Check | Result |
| --- | --- |
| Deterministic formula, binary validation, standard BayesC cadence, fast-block BayesC cadence, and BayesR output tests | 30/30 passed |
| Existing annotated BayesC tests | 106/106 passed |
| Existing annotated BayesR tests | 125/125 passed |
| Full `test/runtests.jl` suite | 1,088/1,088 passed in 3m05.7s |
| Documentation build | passed |

The production-path timing check is recorded in
`benchmarks/reports/2026-08-19-annotation-enrichment-overhead.md`. Across three
1,000-iteration fast-block runs, the median paired overhead was 0.33% on the
local benchmark case.
