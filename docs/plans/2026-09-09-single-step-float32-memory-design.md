# Single-step Float32 memory reduction design

## Problem

The low-memory single-step path currently creates a complete pedigree-ordered
genotype matrix and allocates phenotype- and output-aligned imputed matrices as
`Float64`.  It then converts both aligned matrices to the original genotype
type.  For dense whole-genome analyses, these full animal-by-marker matrices
overlap and dominate peak resident memory.

## Goals

1. Keep the stored imputed genotype matrices in the input genotype precision,
   normally `Float32`.
2. Reorder observed genotypes one marker chunk at a time instead of creating a
   complete pedigree-ordered matrix.
3. Retain `Float64` sparse pedigree solves so the numerical method is not
   weakened by the storage change.
4. Release phase-local matrices explicitly after single-step setup.
5. Preserve the existing `big_memory=true` path and all public APIs.

## Design

Add an internal chunked helper used by `impute_genotypes` when
`big_memory=false`.  The helper will:

- allocate `Mpheno` and `Mout` with `eltype(geno.genotypes)`;
- take a view of at most `markerperchunk` observed-marker columns;
- apply the existing genotype-to-pedigree incidence matrix only to that view;
- run the existing sparse `lhs \\ rhs` pedigree calculation in its promoted
  solve precision;
- copy each solved chunk into the `Float32` destinations, converting once at
  the destination boundary.

The full observed matrix remains available only while chunks are being
imputed.  No complete `Mg` matrix is created in the low-memory path.  At the
end, the genotype object adopts `Mpheno` and `Mout` directly, allowing its old
observed matrix and all local solve matrices to become collectible.

`big_memory=true` retains its whole-matrix calculation, but its final stored
matrices will be converted to the genotype storage type before installation in
the genotype object.

## Numerical behavior

The sparse incidence matrices and pedigree solve retain their current
precision.  Only the destination storage changes: values that were previously
computed in solve precision and converted to `Float32` after constructing the
complete matrix are now converted to `Float32` after each chunk.  The stored
values should therefore agree with the historical low-memory calculation to
`Float32` precision.

## Tests

- Compare the chunked helper with the historical whole-matrix formula on a
  small deterministic problem.
- Exercise more than one marker chunk.
- Verify both returned aligned matrices have the input genotype element type.
- Run the existing single-step tests to cover BayesC, multiple genotype
  categories, annotated BayesC, and annotated BayesR.

## Non-goals

- This change does not add single-step support to the packed streaming backend.
- It does not eliminate the phenotype-aligned genotype matrix required by the
  current dense MCMC sampler.
- It does not change EBV output semantics or remove the output-aligned genotype
  matrix.
