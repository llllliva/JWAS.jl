# Annotation Heritability Enrichment Design

## Goal

Add an opt-in posterior summary of marker-additive heritability enrichment for
each annotation column in annotated BayesC and BayesR models. The implementation
must update once per post-burn-in MCMC iteration. With `fast_blocks`, the main
MCMC iteration is the outer iteration, so block-level inner repetitions must not
be treated as additional posterior draws.

## Interface

Add the `runMCMC` keyword:

```julia
output_annotation_enrichment = false
```

When enabled, every genotype term with annotations returns a table named:

```text
annotation heritability enrichment <genotype name>
```

The usual `runMCMC` output writer will also save that table as a text file.
Unannotated models and runs using the default `false` retain their current
behavior.

## Definition

For marker `j`, trait `t`, annotation `a`, and retained iteration `s`, define:

```math
v_{jt}^{(s)} = 2p_j(1-p_j)(\alpha_{jt}^{(s)})^2
```

```math
V_t^{(s)} = \sum_j v_{jt}^{(s)}, \qquad
V_{at}^{(s)} = \sum_j A_{ja}v_{jt}^{(s)}
```

```math
P_{at}^{(s)} = V_{at}^{(s)}/V_t^{(s)}
```

```math
h_{at}^{2(s)} = V_{at}^{(s)}/(V_t^{(s)}+R_{tt}^{(s)})
```

```math
E_{at}^{(s)} = P_{at}^{(s)}/(M_a/M)
```

Here `M_a` is the post-QC number of markers in annotation `a`, and `M` is
the post-QC total marker count. This definition follows JWAS's existing `2pq`
marker-variance convention and is intentionally a marker-additive summary; it
does not allocate LD covariance.

Annotation columns must be binary when this output is requested. Overlapping
binary annotations are allowed and are interpreted marginally, so their
heritability proportions need not sum to one. The internally added annotation
intercept is not an annotation group and is excluded.

## Sampling and Storage

Initialize a compact accumulator for each annotated genotype term. After marker
effects and the residual variance have been updated, update the accumulator on
every iteration with `iter > burnin`. This location is outside the standard and
fast-block marker kernels and therefore records one draw per full/outer MCMC
state.

Store online first and second moments, the attempted posterior-iteration count,
valid-draw counts, and counts of `enrichment > 1`. Do not store a full
iteration-by-annotation trace. Iterations with zero total sampled marker
variance are skipped and reflected by the difference between
`Posterior_Iterations` and `Posterior_Samples`.

## Initial Scope

- Annotated BayesC and BayesR supported by the existing sampler.
- Single- and multi-trait models, reporting one row per trait and annotation.
- Dense, streaming, ordinary, and fast-block paths where the underlying
  annotated sampler is already supported.
- Random-regression and single-step analyses are rejected for this option in
  the first version because the proposed heritability denominator does not
  include their additional genetic components.

## Verification

Add deterministic tests for the formula and integration tests confirming:

- one summary row per trait and non-intercept annotation;
- one valid draw per post-burn-in outer iteration, independent of output
  thinning;
- finite nonnegative heritability/proportion/enrichment values;
- correct fast-block outer-iteration cadence;
- no enrichment output when the option is disabled;
- clear validation of non-binary annotations and unsupported analysis modes.
