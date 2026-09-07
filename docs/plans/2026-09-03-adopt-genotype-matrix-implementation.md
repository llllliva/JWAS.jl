# Adopt caller-owned genotype matrices: implementation

## Summary

`get_genotypes` now accepts `copy_genotypes::Bool=true`. Existing callers keep
the historical conversion/copy behavior. A caller may opt into ownership
transfer for an already correctly typed dense matrix with
`copy_genotypes=false`.

No-copy mode validates that the input is a dense matrix, storage is `:dense`,
and the matrix element type exactly matches the requested precision. The
matrix is then stored directly in `Genotypes`; centering and imputation may be
visible to the caller. Incompatible precision, DataFrame, file, and streaming
inputs produce clear errors.

## Tests

`test/unit/test_genotype_matrix_ownership.jl` covers default copy identity,
opt-in adoption identity, in-place centering, incompatible precision, and
DataFrame rejection. The focused test passed 8 assertions. The paired
submission-code loader was also smoke-tested with a memory-mappable JLD2
matrix and noncontiguous newID rows.

The complete default test suite (integration tests disabled by its standard
configuration) passed 1,096 of 1,096 assertions on Julia 1.12.5.
