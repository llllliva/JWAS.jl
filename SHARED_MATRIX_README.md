# Shared dense genotype storage

This package adds opt-in `runMCMC(...; shared_genotypes=true)` for dense
single-trait BayesC/BayesR. Training rows, prediction rows and fast SNP blocks
share one dense parent allocation. Existing alignment remains the default.

## Package API

Install dependencies from this checkout with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Supply one already-QCed and imputed `Matrix{Float32}`, with training animals
first and prediction-only animals afterward. Keep phenotype rows and IDs in
matching order; mask prediction-only responses with `missing`.

```julia
using JWAS

# X: Matrix{Float32}; selected_ids: IDs in X row order; snp_ids: column IDs.
# phenotypes: table with ID and y; held-out y entries are missing.
genotypes = get_genotypes(X; method="BayesC", Pi=0.99,
    copy_genotypes=false, quality_control=false, double_precision=false)
genotypes.obsID .= selected_ids
genotypes.markerID .= snp_ids

model = build_model("y = intercept + genotypes")
outputEBV(model, selected_ids)
output = runMCMC(model, phenotypes;
    shared_genotypes=true, double_precision=false,
    fast_blocks=200, chain_length=200000, burnin=500,
    output_samples_frequency=1, output_folder="shared_results")

storage = JWAS.genotype_storage_report(only(model.M))
@assert storage.backing_matrix_count == 1
@assert storage.output_shares_training_parent
@assert storage.all_blocks_share_training
```

Fixed effects and non-marker random effects can be added normally. Annotated
single-trait BayesC/BayesR is supported. Float64 is supported when both import
and MCMC use `double_precision=true`.

`copy_genotypes=false` transfers ownership: preprocessing centers the supplied
matrix in place. Do not modify that matrix afterward. Training and requested
prediction IDs must each describe a nonempty contiguous range in genotype
order. Decode or construct training rows first, rather than permuting an
existing giant matrix. Repeated/permuted rows, single-step, random regression,
non-unit residual weights and unsupported methods fail explicitly.

## Iteration units

With numeric fast blocks this branch divides nominal chain length by block
size. The example runs 1000 outer iterations, discards 500 outer iterations and
saves 500 posterior draws. Inner marker sweeps do not make this equivalent to
200000 ordinary full-model iterations. Check convergence and Monte Carlo
precision on the actual analysis.

## Memory accounting

For A selected animals, P SNPs and block size B, the main Float32 payload is
approximately `4*P*A + 4*P*B` bytes. The second term stores within-block
cross-products. Marker/annotation/model state and Julia overhead still add RAM.
The memory guard accounts for the full retained parent, not only training-view
dimensions. This is a dense in-memory method, not an out-of-core sampler.

`genotype_storage_report` counts actual unique parent matrices and reports
training/output/block sharing and Gram-cache bytes. Disjoint training and
prediction views can share one parent even when their address ranges do not
overlap.

## Verification

The default package suite passed 1776 checks, including 412 shared-matrix
checks. The documentation build passed. Tests covered ordinary/fast BayesC and
BayesR, annotations, several seeds, Float32/Float64 and prediction-only subsets.
The optional simulated-omics integration suite was not enabled.

Eight larger synthetic comparisons used 1200 selected animals, 900 training
animals and 5001 SNPs. Explicit animal and SNP joins verified all expected
keys; EBV, PEV and marker posterior-mean differences were zero. Retained
genotype backing storage fell from 40.06 MiB to 22.89 MiB. These tests establish
implementation agreement, not production convergence, accuracy or peak RSS.

See `docs/src/manual/pic_shared_matrix.md`,
`docs/plans/2026-10-21-pic-shared-matrix-implementation.md`, and
`benchmarks/reports/2026-10-21-pic-shared-matrix/` for details and benchmarks.
Small fast-block view timings were approximately 8-15% slower; benchmark the
memory/runtime tradeoff on production data.

This branch contains the JWAS package, tests, documentation and synthetic
benchmark evidence. Application-specific phenotype preparation, job submission
and runner files are separate from the package.
