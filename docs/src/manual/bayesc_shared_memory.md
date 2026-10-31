# BayesC shared genotype memory

Dense BayesC supports shared genotype storage for single- and multi-trait models,
with ordinary marker updates or fast blocks. Both existing multi-trait samplers
I and II and sequential/independent blocks are supported.

For matching-precision matrix input, BayesC defaults to copy_genotypes=false.
Imputation and centering can modify the input. Use copy_genotypes=true to retain
caller ownership or permit precision conversion. File/DataFrame imports retain
their existing allocation behavior; QC filtering can still allocate a replacement.

BayesC defaults to shared_genotypes=true for one dense genotype term, matching
precision, standard non-single-step/non-RRM analysis, and unit residual weights.
Training and prediction selections must each be contiguous and ordered in
genotype order, without duplicates. Other methods and unsupported model modes
retain upstream copying defaults. Use shared_genotypes=false for noncontiguous
or repeated records. Traits can have different missing phenotypes within the
common training rows; prediction-only animals can have all traits missing.

    # X is Matrix{Float32}; rows are ordered training first, prediction-only last.
    geno = get_genotypes(X, G; method="BayesC", quality_control=false,
        double_precision=false)
    # Set geno.obsID and geno.markerID to the input IDs.
    model = build_model("y1 = intercept + geno\ny2 = intercept + geno", R)
    outputEBV(model, prediction_ids)
    result = runMCMC(model, phenotypes; double_precision=false,
        fast_blocks=100, output_folder="bayesc_shared")
    report = JWAS.genotype_storage_report(only(model.M))
    @assert report.backing_matrix_count == 1

Use fast_blocks=false for the ordinary sampler and multi_trait_sampler=:II
in get_genotypes to select multi-trait sampler II. Set single-/multi-trait
priors G, R and Pi as required by the existing upstream API.

## Residual compatibility and speed

The default fast_residual_updates=false preserves upstream residual arithmetic.
Single-trait and independent multi-trait blocks retain original multiplier types.
Sequential multi-trait blocks reuse a product workspace and copy only the current
alpha block, while retaining the upstream product-then-add evaluation order.

Set fast_residual_updates=true to use in-place precision-matched BLAS residual
updates for BayesC blocks. It can be faster, but different rounding can alter
seeded stochastic trajectories. It is not a bitwise-compatibility mode.
This option is restricted to BayesC and does not affect ordinary marker updates.

Views keep the entire genotype parent alive. Memory reports count unique parents
and block Gram caches; they do not measure peak process RSS or other model state.
The memory guard includes the full retained parent. No out-of-core storage,
prediction formula, prior or sampling-schedule change is introduced.
