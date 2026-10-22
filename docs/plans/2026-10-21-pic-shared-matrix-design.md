# PIC shared genotype matrix: design

Base: llllliva/JWAS.jl, feature/multiclass-fast-blocks,
7e97875ed2bb842d78465cd21e2778aa817d5293 (newest advertised branch tip by
commit date when fetched on 2026-10-21). Existing local checkouts are preserved.
This base already adopts matrices via copy_genotypes=false and stores fast
blocks as views. Plain Matrix training alignment still allocates a second matrix.

Add opt-in runMCMC(...; shared_genotypes=true) for one dense single-trait BayesC
or BayesR genotype term, including annotated variants and non-marker fixed/random
terms. Pair it with get_genotypes(X; copy_genotypes=false, quality_control=false)
using matching Float32 or Float64 precision. Import centering mutates the adopted
matrix; caller ownership transfers to JWAS. Defaults retain existing alignment.

Require ID-checked contiguous ordered training and prediction ranges. Return
UnitRange views of the original selected matrix. Reject arbitrary permutations,
repeated records, duplicate IDs, incompatible precision, single-step, RRM,
non-unit residual weights and unsupported methods, rather than silently copy.
The existing shared multiclass SubArray path remains unchanged when the flag is
false. Do not change priors, marker coding, centering or sampling schedules.

Add a storage report counting unique dense parent buffers and checking aliasing
of training, output and fast-block matrices. In shared mode use actual retained
parent payload in the memory guard, rather than training-view dimensions alone.
Keep the existing fast-block views and AbstractMatrix types.

Update the PIC loader with optional train-first ordering before BED decoding,
preserving old source order by default and keeping all source/newID maps aligned.
Add a shared CLI option and a dedicated direct-bfile wrapper that defaults to
the revised package, train-first order and shared mode. Assert one backing
matrix after production MCMC and save its report; never save a dense genotype
matrix in the direct workflow.

Verify unsupported conditions, Float32/Float64 storage, partial blocks, actual
aliasing, keyed posterior comparisons across several seeds in ordinary/fast
BayesC/BayesR including annotations, plus direct bfile workflows for all four
traits and both FIP time models. Compare the revised shared path with this
same base's copying path, not a different sampler version. Use explicit animal
and marker keys before any numerical comparisons. Record production-path
benchmarks and limits, run the full test suite and attempt the docs build.
