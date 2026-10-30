# Changelog

All notable changes to this project are documented in this file.

## 2026-10-30

### Changed

- Dense matrix input defaults to `copy_genotypes=false`: JWAS adopts a matching-
  precision matrix and may mutate it during imputation and centering. Set
  `copy_genotypes=true` to preserve the input or allow precision conversion.
  File, DataFrame and streaming imports continue allocating their own storage.
- `runMCMC` defaults to `shared_genotypes=true` for genotype analyses. Contiguous
  training, prediction and SNP-block views share one dense parent. The supported
  path is one single-trait BayesC term with matching precision and unit residual
  weights. Other marker models or repeated/permuted/noncontiguous rows require
  `shared_genotypes=false` to use copying alignment. No-marker models are unaffected.
- Block residual updates use precision-matched multipliers for Float32 BLAS
  dispatch. SNP blocks and diagonal products avoid unnecessary matrix/column copies.

## 2026-02-26

### Added

- Low-memory, out-of-core streaming conversion for genotype text files in `prepare_streaming_genotypes`, including staged spool + transpose flow and disk guard checks.
- `conversion_mode` support for `:lowmem`, `:dense`, and `:auto` conversion paths.
- `auto_dense_max_bytes` threshold for automatic conversion-path selection.
- Unit coverage for low-memory conversion behavior, including dense/lowmem auto selection parity.

### Documentation

- Added usage guidance and examples for `conversion_mode` and `auto_dense_max_bytes`.
- Updated streaming backend and large-data walkthrough docs to reflect the low-memory conversion flow.
