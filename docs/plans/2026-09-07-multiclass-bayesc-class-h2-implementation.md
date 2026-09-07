# Multiclass BayesC class heritability implementation

## Delivered behavior

- Single-trait dense models containing two or more BayesC genotype terms now
  write `MCMC_samples_class_heritability.txt` when
  `output_heritability=true`.
- Columns use the genotype-term/class names and rows follow the existing
  burn-in and `output_samples_frequency` schedule.
- The calculation does not require `outputEBV=true`.
- Each class receives `Cov(g_c, g_total) / (Var(g_total) + residual_variance)`.
  Consequently, the class values sum to total marker heritability and retain
  cross-class covariance exactly once. Negative contributions are preserved.
- The calculation operates on aligned class matrix views and does not copy
  genotype matrices.

## Validation

- Focused shared dense multiclass suites: 13/13 construction tests and 11/11
  MCMC/class-heritability tests passed.
- Full JWAS unit suite: 1,162/1,162 passed on Julia 1.10.2.
- Documenter manual build completed successfully; deployment was skipped
  outside CI because the copied folder has no Git metadata.

## Benchmark

The shared dense production-path benchmark used 100 observations, 1,000 base
SNPs, 1,668 class memberships, and three classes. Warm class-heritability
calculation allocation was 7,040 bytes. All class genotype matrices continued
to alias the base matrix. The expected leading temporary memory is
approximately `8 * nobs * (nclasses + 1)` bytes.
