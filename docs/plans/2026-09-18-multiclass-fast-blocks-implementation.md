# Multiclass fast blocks implementation

Base: `33c37f0` on `feature/shared-dense-multiclass-h2`.
Branch: `feature/multiclass-fast-blocks`.

Numeric and automatic fast-block settings now resolve a block-start vector for
each genotype term. Gibbs matrix construction and memory estimates select the
same local starts. The first class no longer determines every class's marker
coverage. Single-block classes are supported in multiclass models. Explicit
vectors are validated against every term, rejecting invalid starts before block
construction. Numeric schedules reject an outer chain length at or below burn-in.

The existing sequential BayesABC block kernel and shared residual are unchanged.
Class priors, overlapping effects, variance updates, class heritability and
inner/outer statistics remain separate. Genotype block views retain shared
storage. Numeric/automatic chain conversion and block-width inner repetitions
are preserved, and documented with a 140,000 / 100 = 1,400 outer-iteration example.

Regression coverage includes automatic/numeric/explicit starts; unequal classes
of 3, 5 and 1 markers; overlap; noncontiguous source columns and phenotype rows;
unit and nonunit residual weights; blocks larger than a class; full marker
coverage and shared memory; per-class inner sample counts and PIP bounds; and
residual reconstruction after every sequential class update with nonzero initial
effects. Validation on Julia 1.11.7:

- `julia --project=. --startup-file=no test/runtests.jl`: 1,364 / 1,364 passed
  (default suite; optional integration tests disabled).
- Final focused `test/unit/test_shared_dense_multiclass.jl`: 213 / 213 passed.
  The focused rerun includes the final nonmonomorphic singleton fixture.
- `julia --project=docs --startup-file=no docs/make.jl`: passed after installing
  the recorded documentation dependencies; local deployment skipped normally.
- `git diff --check`: passed.

No production-scale runtime or posterior-mixing benchmark was performed. The
unit MCMC runs verify execution and bookkeeping, not Monte Carlo convergence.
