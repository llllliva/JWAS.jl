# One shared dense genotype matrix

For single-trait BayesC/BayesR with one dense genotype term, opt in to
`runMCMC(...; shared_genotypes=true)`. Genotypes are held once for selected
training and prediction animals. Training rows and fast SNP blocks refer to
that parent allocation through views, while posterior EBVs and PEVs are still
calculated for all requested animals.

```julia
using JWAS

# X is one already-imputed, already-QCed Matrix{Float32}.
# Decode training rows first, then test rows, before allocating X.
genotypes = get_genotypes(X; method="BayesC", Pi=0.99,
    copy_genotypes=false, quality_control=false, double_precision=false)
genotypes.obsID .= selected_ids  # same order as X and the phenotype table
genotypes.markerID .= snp_ids

model = build_model("y = intercept + FARM + LITTER + genotypes")
set_random(model,"LITTER")
outputEBV(model,selected_ids)

# ph.y is observed for training and missing for test animals.
result = runMCMC(model,ph; shared_genotypes=true, double_precision=false,
    fast_blocks=100, chain_length=200000, burnin=400,
    output_samples_frequency=5, output_folder="new_shared_results")

report = JWAS.genotype_storage_report(only(model.M))
@assert report.backing_matrix_count == 1
@assert report.output_shares_training_parent
@assert report.all_blocks_share_training
```

`copy_genotypes=false` transfers ownership of X to JWAS. Centering mutates it
in place. Do not modify X afterward. Use matching precision at import and MCMC;
Float64 is also supported when both `double_precision` flags are true. The
shared alignment keyword defaults to false, preserving existing alignment.

Training IDs must select one nonempty contiguous range in genotype order.
Prediction IDs may select the full matrix or one contiguous range. Arbitrary
permutations and repeated records are rejected explicitly. Decode train-first
from BED rather than permuting an existing giant matrix. Original/new ID maps
must follow the chosen decoded row order.

The initial scope is one dense genotype term, a single trait, ordinary or exact
sequential fast-block BayesC/BayesR, with or without marker annotations and with
ordinary categorical/covariate/non-marker random terms. Single-step, RRM,
non-unit residual weights and other methods are rejected. The existing shared
multiclass API is separate and retains its previous behavior.

The large genotype payload is approximately `4 * selected_animals * SNPs` bytes
for Float32. Fast blocks add about `4 * SNPs * block_size` bytes for their
cross-products, plus other model, marker, annotation and Julia allocations.
The memory guard includes the retained full parent, not just training-view
dimensions. A storage report counts actual unique dense parent buffers and
reports aliasing and block-Gram payload. This is a dense in-memory method;
full production peak RSS still needs measurement on a compute-node pilot.

Numeric fast blocks divide nominal chain length by block size in this branch.
The example runs 2,000 outer iterations, discards 400 outer burn-in iterations,
and saves 320 draws. These example settings do not establish convergence.

Application code is responsible for constructing the selected matrix and
matching ID order before calling this package API. No application-specific
phenotype loader or job runner is included in this branch.
