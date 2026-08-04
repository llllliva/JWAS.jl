# BayesR Collapsed Pi AlphaSimR Comparison

## Question

Does integrating the global BayesR `pi` vector out of marker-class updates
reduce short-chain PIP inflation without increasing runtime or damaging
prediction?

## Design

The production JWAS path was run on both retained AlphaSimR architectures:

- `small_effect_polygenic`: 500 causal markers;
- `sparse_large_effect`: 10 causal markers;
- 1,000 training individuals, 200 test individuals, and 10,000 markers;
- two seeds: `20260803` and `20260804`;
- 300 outer iterations, 30 burn-in iterations, output every 5 iterations;
- 54 retained outer samples per run;
- starting `Pi = [0.94, 0.049, 0.01, 0.001]`;
- `BayesR_gamma = [0.0, 0.1, 1.0, 10.0]`;
- estimated `Pi` and shared marker variance;
- dense conventional marker sweeps; and
- marker-effect sample matrices disabled.

All truth and pairwise comparisons joined results by explicit `Marker_ID`.
Marker counts, unique IDs, marker-key sets, join row counts, and test-individual
ID sets were asserted before summaries were calculated.

## Results

Values below are means over the two seeds.

| Architecture | Sampler | Mean PIP | Causal PIP | Null PIP | PIP/truth correlation | Phenotype prediction correlation | Genetic-value prediction correlation |
|---|---|---:|---:|---:|---:|---:|---:|
| 500 small effects | standard | 0.12833 | 0.13328 | 0.12807 | 0.0227 | 0.6822 | 0.8405 |
| 500 small effects | collapsed | 0.12415 | 0.12704 | 0.12400 | 0.0140 | 0.6925 | 0.8439 |
| 10 sparse effects | standard | 0.01252 | 0.33148 | 0.01220 | 0.3120 | 0.7389 | 0.9782 |
| 10 sparse effects | collapsed | 0.00942 | 0.31111 | 0.00912 | 0.2965 | 0.7342 | 0.9754 |

For the small-effect architecture, collapsing reduced mean PIP by 3.25% and
mean null-marker PIP by 3.18%. For the sparse architecture, it reduced mean PIP
by 24.77% and mean null-marker PIP by 25.28%.

The sparse improvement was not free: average causal-marker PIP decreased from
0.3315 to 0.3111, and PIP/truth correlation decreased slightly. Prediction was
nearly unchanged in both architectures.

After compilation, elapsed time was similar. The second small-effect run took
1.28 seconds with the standard sampler and 1.25 seconds collapsed. Sparse runs
were approximately 1.16 seconds standard and 1.11 seconds collapsed at the
median. The first standard run included about nine seconds of compilation and
is not an algorithmic timing comparison.

## Mixing Evidence

The 500-effect result remained strongly seed-sensitive:

| Sampler | Seed 20260803 mean PIP | Seed 20260804 mean PIP |
|---|---:|---:|
| standard | 0.15505 | 0.10161 |
| collapsed | 0.14931 | 0.09900 |

Thus collapsing did not resolve short-chain mixing. It modestly moved both
chains toward sparser occupancy but did not make them agree. Marker-level
standard-versus-collapsed PIP correlations were only 0.067 and 0.126 for this
architecture.

For the sparse architecture, standard-versus-collapsed marker PIP correlations
were higher, 0.586 and 0.479, but top-100 overlaps were still only 14 and 10.
High LD and only 54 retained outer samples remain important limitations.

## Interpretation

The collapsed sampler is computationally inexpensive and can reduce dense
occupancy in a short run, particularly for a sparse architecture. However, with
the same `Dirichlet(1,1,1,1)` prior it targets the same posterior as the standard
sampler. Differences at 300 iterations therefore reflect transition and
initialization behavior rather than a principled change in sparsity.

Collapsed runs initialize class counts from the supplied starting `Pi`, whereas
the preserved standard JWAS path initializes all class labels to null and then
uses starting `Pi` in its first marker sweep. This necessary difference should
also be remembered when interpreting very short chains.

## Conclusion

Keep `:standard` as the default and expose `:collapsed` as an experimental,
opt-in mixing alternative. The present evidence supports using it in sensitivity
analysis, especially for sparse traits, but not using it alone as the PIP
inflation correction. Sparse-prior sensitivity, more retained global updates,
and between-chain agreement remain necessary.

Machine-readable results are in:

```text
Test_samples/JWASvsBayesR3/AlphaSimR_smallNe_architecture/
  jwas_collapsed_pi_comparison/run_summary.csv
  jwas_collapsed_pi_comparison/pairwise_summary.csv
```
