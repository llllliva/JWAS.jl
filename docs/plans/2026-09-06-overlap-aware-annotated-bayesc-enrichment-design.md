# Overlap-aware annotated BayesC enrichment design

## Problem

The existing annotated enrichment calculation multiplies every marker variance
by every binary annotation membership. A marker in two annotations is therefore
counted twice. Annotation heritability/proportion totals are not conserved, and
the interpretation differs from the variance partition used for multiclass
BayesC.

## Scope

Change the enrichment calculation for single-trait annotated BayesC. Preserve
the sampler, marker effects, marker-specific pi update, ordinary BayesC, BayesR,
and multi-trait behavior. The change affects reporting/accumulation only; it does
not change the posterior model being sampled.

## Allocation rule

For marker `j`, let

```math
v_j = 2p_j(1-p_j)\alpha_j^2
```

and let `S_j` be its active annotation columns. The current annotated BayesC
linear predictor is

```math
\eta_j = \gamma_0 + \sum_{k \in S_j}\gamma_k,
```

with inclusion probability `Phi(eta_j)`. For each active annotation, define its
leave-one-out probability contribution

```math
s_{jk} = |\Phi(\eta_j)-\Phi(\eta_j-\gamma_k)|.
```

Normalize within marker:

```math
q_{jk} = s_{jk}/\sum_{l \in S_j}s_{jl}.
```

If all scores are numerically zero, use `q_jk = 1/|S_j|`. An unannotated marker
gets `q=1` in an automatically reported `Unannotated` group. The allocated
annotation variance is

```math
V_k = \sum_j q_{jk}v_j.
```

Consequently, every marker variance is allocated exactly once and
`sum(V_k) == sum(v_j)` up to floating-point error.

## Enrichment baseline

The baseline SNP share must not depend on posterior annotation coefficients.
Use equal fractional membership for this denominator: an annotated marker
contributes `1/|S_j|` to each active annotation, and an unannotated marker
contributes one to `Unannotated`. Divide these effective counts by the total
marker count to obtain fixed fractions `f_k`. Then

```math
H_k^2 = V_k / (\sum_j v_j + \sigma_e^2),
```

```math
P_k = V_k / \sum_j v_j,
```

```math
E_k = P_k/f_k.
```

This gives `sum(P_k)=1`, `sum(H_k^2)=H_marker^2`, and
`sum(f_k E_k)=1` for every valid draw.

## Performance and storage

Do not materialize an `nmarkers by nannotations` allocation matrix. Reuse the
existing annotation design and accumulate into one vector of length
`nannotations + has_unannotated`. Runtime scans the annotation design and
evaluates probit contributions only for active memberships; additional
persistent memory is linear in the number of reported groups.

## Tests

- Verify nonuniform allocation from known probit coefficients.
- Verify equal splitting when all leave-one-out scores are zero.
- Verify the automatic `Unannotated` group.
- Verify variance, heritability, and enrichment conservation identities.
- Preserve MCMC cadence/output tests and non-BayesC compatibility tests.
