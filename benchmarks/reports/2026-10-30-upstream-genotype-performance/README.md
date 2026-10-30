# Compatibility with upstream JWAS 2.4.0

Base: `reworkhow/JWAS.jl` master
`038c6c8a497ec6e17e957657a5c13ae698377132`, fetched 2026-10-30.
Local runtime: Julia 1.11.7, Apple Silicon, one default Julia thread. Runs used
`--compiled-modules=existing`, `--startup-file=no`, and `GKSwstype=100`.

## Regression results

| Run | Result | Interpretation |
| --- | --- | --- |
| Untouched upstream source and tests | 988 passed | Baseline is healthy in this environment. |
| Port with unchanged upstream tests | 779 passed, 5 failed, 42 errored | The requested defaults are intentionally not backward compatible. Shared-mode validation rejects formerly valid unsupported analyses and changes which validation message appears first. |
| Port with explicit legacy copying opt-outs and new coverage | 2352 passed | Existing paths work with opt-outs; supported sharing and new defaults pass their own tests. |
| Documentation build | Passed | Built from the port source; no deployment. |

Existing regression cases explicitly set `shared_genotypes=false` and
`copy_genotypes=true`. File/DataFrame ownership flags have no effect; their
imports retain upstream allocation behavior. No existing assertion was removed.
New coverage tests omitted keywords, caller mutation and precision conversion,
contiguous row guards, shared parent identity, annotations, Float32/Float64,
BayesC/BayesR, sequential/independent blocks, and residual reconstruction.
Optional simulated-omics integration tests and other Julia/OS combinations
were not run.

## Numerical and storage comparisons

`storage.csv` summarizes eight synthetic copied-versus-shared production runs:
BayesC/BayesR, seeds 73/123, ordinary or size-20 blocks, 128 selected animals,
96 training animals and 257 markers. Fixed FARM and random LITTER effects are
included. Retained dense genotype payload fell from 230272 to 131584 bytes
(42.9%); unique parents fell from two to one. Copied and shared outputs agree
exactly in these comparisons. Both paths use the new block views. These byte
counts exclude Gram caches, other model state and runtime overhead, and are
not peak RSS.

`upstream_output_comparison.csv` compares the port's defaults directly with
untouched upstream using identical synthetic inputs, seeds, and upstream's
block schedule. Ordinary-mode EBV/PEV and marker summaries matched exactly.
Fast-block maximum absolute differences across four comparisons were:

- EBV: 4.046e-6
- PEV: 9.136e-7
- marker posterior mean: 1.660e-7
- marker posterior SD: 1.540e-7
- model frequency: 0

All values pass `atol=1e-4`, `rtol=1e-4`. Precision-matched Float32 BLAS
accumulation need not be bitwise identical to the old generic kernel.
Each run uses 300 outer iterations, discards 100 and saves 20 draws. These
runs establish implementation agreement, not convergence or production accuracy.

Comparisons assert expected animal/marker counts and unique matching key sets
before joining by ID and Trait/Marker_ID. Quoting is normalized in marker keys;
comparison-ready aligned tables are generated locally and ignored in Git.
Joined sample rows were inspected.

## Reproduction

From the port root:

```sh
julia --project=. --startup-file=no test/runtests.jl
julia --project=. --startup-file=no benchmarks/upstream_genotype_storage.jl
# Run the same exporter against separate untouched-upstream and port projects:
julia --project=/path/to/upstream --startup-file=no benchmarks/upstream_genotype_output_compatibility.jl /tmp/output-upstream
julia --project=. --startup-file=no benchmarks/upstream_genotype_output_compatibility.jl /tmp/output-port
python3 benchmarks/compare_upstream_genotype_outputs.py /tmp/output-upstream /tmp/output-port benchmarks/reports/2026-10-30-upstream-genotype-performance
```

The documentation build used Documenter from an isolated environment with the
port project first on JULIA_LOAD_PATH; the loaded JWAS path was verified.
Upstream Project.toml, Manifest.toml and package version remain unchanged.
