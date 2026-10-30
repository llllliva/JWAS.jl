# Genotype memory and residual performance

Start from origin/master (a619a30222b884f9fc0a6030dde375de35c9f841).
Include only dense matrix adoption, contiguous genotype row/block views,
and precision-matched block residual updates. Use `copy_genotypes=false` and `shared_genotypes=true` by default.
Explicit opt-outs retain copying. File/DataFrame parsing
and streaming imports ignore the matrix-ownership option. Shared alignment
constraints now apply by default to marker analyses; conventional no-marker
analyses have no genotype storage to share.
Do not introduce annotation models, BayesR, multiclass models, sampling schedules,
or independent-block sampling absent from this base.

No-copy input requires a dense Matrix with matching precision and transfers
preprocessing mutation to the caller. Shared alignment requires one dense
single-trait BayesC term, contiguous unique row selections, unit residual
weights, and matching import/MCMC precision; unsupported cases fail explicitly.
Keep the retained parent matrix in memory estimates.

Validation: ownership identity and mutation, shared-parent identity and block
views, keyed production output comparisons across seeds/precisions, residual
reconstruction and full existing regression suite. Run documentation build.
Measure allocation and residual-kernel behavior without claiming full-analysis
speedups from a kernel measurement.
