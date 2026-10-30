# Implementation

Base: llllliva/JWAS.jl master a619a30222b884f9fc0a6030dde375de35c9f841.

The focused changes add dense matrix adoption, contiguous row/block
views, and precision-matched BayesABC block residual multipliers. Genotypes and
GibbsMats accept floating-point matrix views. Block containers retain SubArray
objects instead of converting them to Matrix, and diagonal products use
column views. Shared alignment retains a common training/prediction parent;
the guard counts the retained parent beyond the training-view payload.
The defaults are
`copy_genotypes=false` and `shared_genotypes=true`. Explicit opt-outs retain
the copying paths. File/DataFrame/streaming imports ignore matrix ownership;
no-marker conventional models skip sharing validation. Matching-precision
matrix inputs can be mutated during preprocessing by default. Marker analyses
outside shared-mode support require `shared_genotypes=false`. Legacy regression
cases explicitly use copying alignment to retain their established test semantics;
ownership and production-sharing tests exercise the new defaults.

This base has no BayesR or independent-block sampler; neither is introduced.
Shared mode supports one dense single-trait BayesC term and rejects unsupported
row layouts, methods, precision, weights and model modes. Annotation and block
sampling algorithms and schedules are unchanged.

Validation: the full suite passed 1061 checks with the new defaults using
Julia 1.11.7, `--compiled-modules=existing`, `--startup-file=no`, and
`GKSwstype=100`. Tests include omitted-keyword ownership and shared alignment,
explicit copying opt-outs, expected animal/marker counts and keyed comparisons,
and conventional no-marker default behavior. Optional simulated-omics
integration tests were not enabled. Documentation built successfully using the
isolated Documenter environment; deployment was skipped by Documenter.

Synthetic measurements and their limits are documented under
`benchmarks/reports/2026-10-30-genotype-performance/`. No production data or
application launchers are included.
