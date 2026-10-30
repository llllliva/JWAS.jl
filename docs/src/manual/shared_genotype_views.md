# Shared genotype storage

For a dense, single-trait BayesC analysis, opt in to one genotype allocation
shared by training rows, prediction rows, and SNP blocks:

```julia
geno = get_genotypes(X, 1.0; method="BayesC", Pi=0.95,
    copy_genotypes=false, quality_control=false, double_precision=false)
# X must be Matrix{Float32}; set geno.obsID and geno.markerID in X order.
model = build_model("y = intercept + geno", 1.0)
outputEBV(model, geno.obsID)
runMCMC(model, phenotypes; shared_genotypes=true, double_precision=false,
    fast_blocks=200, output_folder="shared_results")
report = JWAS.genotype_storage_report(only(model.M))
@assert report.backing_matrix_count == 1
```

`copy_genotypes=false` transfers preprocessing ownership: imputation and
centering can modify `X` in place. Do not mutate it while the model uses it.
With QC enabled, marker filtering can still allocate a replacement matrix.
Both options default to the existing copying behavior.

Training and output IDs must each form a nonempty contiguous range in genotype
order. Construct training rows first and prediction-only rows afterward; keep
phenotype IDs aligned and mark held-out responses `missing`. Repeated,
permuted, or noncontiguous selections fail explicitly. Disjoint training/output
views still retain the same parent matrix.

Matching Float64 import and MCMC precision is also supported. Shared mode rejects
multiple genotype terms, multiple traits, relationship matrices, single-step,
random regression, streaming storage, other methods, and non-unit residual
weights. These restrictions apply only when `shared_genotypes=true`.

The memory guard includes the full retained dense parent and block Gram caches.
Views do not provide out-of-core storage. `genotype_storage_report` counts unique
parent buffers rather than overlapping address ranges; its byte counts exclude
other model state and do not measure peak RSS.

Block residual updates use multipliers matching the residual element type,
allowing Float32 BLAS matrix-vector dispatch. This does not change the block
sampling schedule. Kernel timings do not establish whole-analysis speedups.
