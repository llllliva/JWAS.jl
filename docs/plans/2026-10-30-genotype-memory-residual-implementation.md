# Implementation

Base: llllliva/JWAS.jl master a619a30222b884f9fc0a6030dde375de35c9f841.

Three focused commits add opt-in dense matrix adoption, contiguous row/block
views, and precision-matched BayesABC block residual multipliers. Genotypes and
GibbsMats accept floating-point matrix views. Block containers retain SubArray
objects instead of converting them to Matrix, and diagonal products use
column views. Shared alignment retains a common training/prediction parent;
the guard counts the retained parent beyond the training-view payload.
Defaults preserve the copying alignment and matrix-input ownership behavior.

This base has no BayesR or independent-block sampler; neither is introduced.
Shared mode supports one dense single-trait BayesC term and rejects unsupported
row layouts, methods, precision, weights and model modes. Annotation and block
sampling algorithms and schedules are unchanged.

Validation: default full suite passed 925 checks with Julia 1.11.7 using
`--compiled-modules=existing`, `--startup-file=no`, and `GKSwstype=100`. Optional
simulated-omics integration tests were not enabled. Documentation built
successfully using the isolated Documenter environment; deployment was skipped
by Documenter. Focused tests passed.
Additional explicit expected animal/marker-key assertions were subsequently
added and checked with the focused suite.

Synthetic measurements and their limits are documented under
`benchmarks/reports/2026-10-30-genotype-performance/`. No production data or
application launchers are included.
