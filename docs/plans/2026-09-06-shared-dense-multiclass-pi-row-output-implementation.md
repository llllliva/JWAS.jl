# Shared dense multiclass and row-oriented annotated BayesC pi output: implementation

## Delivered changes

- Added `get_shared_genotype_classes(source, memberships; ...)`. Each class owns its
  BayesC parameters and MCMC state, but its genotype matrix is a column view into one
  dense, post-QC `Genotypes` object. Overlapping SNPs therefore share genotype storage
  instead of copying genotype columns into every class.
- Preserved these shared views when phenotype IDs require genotype-row reordering.
- Updated dense Gibbs helpers to accept `AbstractMatrix` inputs and to compute column
  norms through views.
- Added an optional `Others` class for SNPs having no annotation membership.
- Changed single-trait annotated BayesC per-SNP pi output so the header contains marker
  IDs and every retained MCMC iteration is one row. Ordinary scalar BayesC, BayesR,
  and multi-trait output formats are unchanged.
- Added a manual page, README example, focused unit/MCMC tests, and a memory benchmark.
- Set `remotes = nothing` in the copied package's Documenter configuration so its docs
  can build even though the copied directory intentionally has no Git metadata.

## Validation

- Full unit test suite: 1,116 passed, 0 failed on Julia 1.10.2.
- Documentation: built successfully with Documenter; deployment was correctly skipped
  outside CI.
- Focused tests verify overlapping marker membership, the `Others` class, memory
  aliasing before and after model alignment/MCMC, independent per-class state, invalid
  membership rejection, annotated pi dimensions/header, and compatibility of ordinary
  scalar BayesC output.

## Benchmark result

The production-path benchmark used 100 observations, 1,000 base SNPs, and 1,668 total
class memberships. The base `Float32` genotype values occupied 400,000 bytes. Physical
copies for the class matrices would require 667,200 bytes, while warm construction of
the shared class wrappers allocated 56,192 bytes and every class continued to alias the
base matrix. A two-iteration MCMC smoke run also completed.

For a 300 GB base matrix and the observed annotation membership ratio of 2.5457, copied
class genotype values alone would be about 764 GB. The shared implementation retains
one 300 GB genotype matrix. This does not eliminate membership vectors, class-specific
effects, residual work arrays, block matrices, sampler state, or Julia/BLAS overhead.

## Operational limits

- The helper accepts a dense BayesC source object. Build and QC the source once, then
  pass only the returned class objects to the model.
- Membership rows must correspond to the source object's post-QC marker order and must
  be binary. Overlap is allowed.
- A SNP in two classes has two class-specific regression coefficients and contributes
  through both model terms. Shared design storage changes memory ownership, not the
  statistical meaning or overlap identifiability of multiclass BayesC.
- Row-oriented per-SNP pi samples can be extremely large. With 1,385,033 SNPs, a chain
  length of 250,000, burn-in 600, and output frequency 1, the output contains
  249,400 rows by 1,385,033 columns (345,427,230,200 numeric cells), likely several TB
  as delimited text. Use an appropriate output frequency/thinning policy for production.
