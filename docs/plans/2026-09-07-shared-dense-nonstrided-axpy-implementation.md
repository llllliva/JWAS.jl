# Shared-dense non-strided BayesC residual update implementation

The BayesC marker update now dispatches residual AXPY operations to BLAS for
strided vectors and to an allocation-free SIMD loop for vector-indexed shared
genotype views. This preserves the fast path for ordinary dense matrices while
allowing shared-dense multiclass models to align a phenotype subset without
copying genotype columns.

A regression test runs three shared classes, including an unassigned class,
with two genotype rows absent from the phenotype table. The focused shared
dense suite passed 28/28 assertions, both submission-runner smoke workflows
completed, and the full package suite passed 1,179/1,179 tests.
