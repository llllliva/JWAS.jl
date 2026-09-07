# Shared-dense non-strided BayesC residual update fix

## Problem

Shared-dense multiclass terms use zero-copy column views. When the phenotype
training set is a subset of genotype IDs, alignment adds a vector-indexed row
view. A marker column from that nested view is not a `StridedVector`, so
`BLAS.axpy!` throws before the first included-marker residual update.

## Change

Keep the BLAS path for strided marker columns and use an allocation-free scalar
AXPY loop for non-strided views. Add a shared-dense multiclass regression test
with unphenotyped genotype rows and verify that MCMC and class-heritability
output complete.
