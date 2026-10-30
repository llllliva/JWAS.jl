# Genotype storage and residual-kernel measurements

Measured locally with Julia 1.11.7 on Apple Silicon. Run
`julia --project=. --startup-file=no benchmarks/genotype_memory_residual.jl`
from the repository root. The local run used `--compiled-modules=existing` and
`GKSwstype=100` to avoid cache writes and graphics initialization.

The production BayesC comparisons use 128 selected animals, 96 training animals,
257 markers, seeds 73 and 123, and ordinary or size-20 blocks. Each run retains
200 post-burn-in outer iterations and saves 20 draws; these runs establish
implementation agreement rather than convergence. Fixed FARM and random LITTER
effects are included.

The benchmark asserts expected animal/marker counts, unique keys and equal key
sets, then joins by ID and by Trait/Marker_ID. It writes aligned comparison
tables locally (ignored in Git). `storage.csv` contains the summarized results.
All four comparisons had zero EBV and marker posterior-mean differences.
Retained genotype backing payload fell from 230272 to 131584 bytes (42.9%),
from two parents to one. This excludes Gram caches, other model state, and
Julia overhead; it is not a peak-RSS measurement or a comparison to the old
block-materializing implementation. Both paths here use the new block views.

`residual_kernel.csv` compares `mul!` with Float64 literals against multipliers
matching the residual precision. It uses a 3072-by-200 view of a 4096-by-200
parent, one BLAS thread, warm-up, and the median of seven batches of 20 calls.
The Float32 kernel measured 459.1 microseconds versus 51.5 microseconds (8.92x);
Float64 was effectively unchanged (144.8 versus 144.1 microseconds). The
numerical results pass an approximate-equality check. These are local kernel
measurements, not whole-chain or production-data speedup claims.

The full package suite passed 1061 checks with the new defaults. Focused tests also cover
Float32/Float64 residual reconstruction, matrix ownership, contiguous selection
guards, shared block parents, annotations supported by the base, and production
output agreement across seeds 73/123/456 and both precisions.
