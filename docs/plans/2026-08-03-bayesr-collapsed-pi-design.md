# BayesR Collapsed Pi Sampler Design

## Goal

Add an opt-in collapsed sampler for the ordinary single-trait BayesR mixture
proportions while preserving the existing sampled-`pi` Gibbs transition as the
default.

## Public API

`runMCMC` gains:

```julia
BayesR_pi_sampler = :standard
```

Accepted values are `:standard` and `:collapsed` (strings with the same names
are also accepted). `:standard` preserves the current behavior. `:collapsed`
integrates the global four-class `pi` vector out of each marker-class update.

The collapsed option requires:

- an ordinary single-trait BayesR genotype term;
- `estimatePi=true`;
- no marker annotations;
- dense genotype storage; and
- `independent_blocks=false`.

Both conventional marker sweeps and sequential fast-block sweeps are supported.

## Statistical Kernel

Let `delta[j]` be marker `j`'s current class, `n[-j,k]` the number of other
markers in class `k`, and `L[j,k]` the existing BayesR conditional likelihood.
The standard update uses

```text
Pr(delta[j] = k | pi, ...) proportional to L[j,k] * pi[k].
```

For the existing `Dirichlet(1,1,1,1)` prior, the collapsed update uses

```text
Pr(delta[j] = k | delta[-j], ...) proportional to
    L[j,k] * (n[-j,k] + 1).
```

Before calculating the weights, the marker's old class is removed from the
global counts. Its newly sampled class is then added. This exclusion is required
for the exact full conditional.

The existing end-of-outer-cycle draw

```text
pi ~ Dirichlet(class_counts + 1)
```

is retained for posterior output. Under `:collapsed`, that draw is not used by
the following marker sweep. Thus `pi` output remains available without
reintroducing sampled-`pi` dependence into marker allocations.

Collapsed runs initialize class assignments from the supplied starting `Pi`.
This supplies a usable initial count state without treating `Pi` as an
additional prior. Standard runs retain their existing initialization exactly.

## Compatibility and Limitations

- The default remains `:standard`; existing random-number streams and results
  must not change when the new keyword is omitted.
- Annotated BayesR has marker-specific mixture probabilities and is not covered
  by the simple global Dirichlet collapse.
- Independent blocks update in parallel from a shared residual snapshot. A
  mutable global class-count vector would make that kernel order-dependent and
  unsafe, so this combination is rejected.
- Collapsing changes the transition kernel but not the target posterior. It can
  improve short-chain mixing, but it is not a new sparsity prior and is not
  guaranteed to reduce a genuinely dense posterior.

## Verification

1. Unit-test option parsing and invalid combinations.
2. Verify explicit `:standard` and the omitted keyword give identical seeded
   low-level transitions.
3. Verify collapsed class counts remain valid and are used after excluding the
   current marker.
4. Run focused BayesR tests and the full package test suite.
5. Run matched production-path simulations for the AlphaSimR
   `small_effect_polygenic` and `sparse_large_effect` architectures, comparing
   standard and collapsed samplers by explicit marker IDs and reporting PIP,
   class proportions, variance summaries, prediction metrics, and runtime.
