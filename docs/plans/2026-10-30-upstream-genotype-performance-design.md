# Genotype performance port to current upstream

Base: reworkhow/JWAS.jl master 038c6c8a497ec6e17e957657a5c13ae698377132,
JWAS 2.4.0, fetched 2026-10-30. Port only dense matrix adoption, contiguous
training/prediction/block views, and precision-matched residual multipliers.
Defaults: copy_genotypes=false and shared_genotypes=true. Preserve copying
opt-outs and upstream annotation algorithms and sampling schedules.

Unlike the previous fork base, current upstream includes single-trait BayesR
and sequential/independent BayesC blocks. Cover those existing paths without
importing unrelated code from the fork. Shared mode supports one dense
single-trait BayesC/BayesR term with matching precision, contiguous unique rows
and unit weights. Other marker models/layouts require shared_genotypes=false.
Matrix inputs require matching precision for adoption, or copy_genotypes=true
for conversion. File/DataFrame/streaming import ownership stays unchanged.

Check baseline upstream separately. Validate the port with upstream's full
regression suite, explicit opt-outs for legacy unsupported paths, omitted-keyword
default tests, both precisions and block modes, keyed output comparisons across
seeds, storage identity, retained-parent accounting, and documentation build.
Compatibility includes documenting intentional default-related API breaks.
Do not submit an upstream PR or modify either master branch during review.
