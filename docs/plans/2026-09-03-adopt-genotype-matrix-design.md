# Adopt caller-owned genotype matrices without copying

## Motivation

Whole-genome JLD2 inputs can be hundreds of gigabytes. A runner can
memory-map the source file and materialize only selected training and testing
rows, but `get_genotypes` currently copies that already-selected `Float32`
matrix. The duplicate raises peak memory without changing values.

## Design

Add an opt-in `copy_genotypes` keyword to `get_genotypes`.

- `copy_genotypes=true` remains the default and preserves current behavior.
- `copy_genotypes=false` is accepted only for concrete numeric matrices whose
  element type already matches `double_precision`.
- In no-copy mode, JWAS adopts the caller's matrix. Centering, missing-value
  imputation, and other in-place processing may therefore mutate it.
- Inputs requiring precision conversion continue to require copying and
  produce a clear error in no-copy mode.
- DataFrame and file inputs retain their existing behavior.

## Verification

Add unit coverage showing that default matrix input is copied, opt-in input is
adopted by identity, in-place centering is visible to the caller, and an
incompatible element type is rejected.

