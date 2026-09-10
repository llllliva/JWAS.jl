# Single-step Float32 memory reduction implementation

## Delivered changes

- Added `_impute_genotypes_in_chunks`, an internal helper that applies the
  genotype-to-pedigree reorder to at most 1,000 markers at a time.
- Allocated phenotype- and output-aligned genotype matrices directly with the
  observed genotype element type (`Float32` in the production workflow).
- Kept sparse pedigree calculations in their promoted solve precision and
  converted once when writing each result chunk into final storage.
- Installed the completed matrices directly in the `Genotypes` object.  This
  replaces its reference to the observed matrix and avoids complete
  post-imputation conversion copies.
- Preserved the existing `big_memory=true` whole-matrix path and public API.
- Added deterministic parity/type tests and a repeatable peak-RSS benchmark.

## Verification

- Focused single-step tests: 40/40 assertions passed, including the new
  three-chunk parity test plus BayesC, annotated BayesC, annotated BayesR, and
  multiple genotype categories.
- Full package suite: 1,183/1,183 tests passed on Julia 1.10.2; integration
  tests remained disabled according to the suite's default configuration.
- `git diff --check`: passed.

## Benchmark

The 4,000-genotyped-animal by 10,000-marker benchmark produced identical
checksums.  Peak RSS decreased from 1.782 GiB to 1.148 GiB, a reduction of
0.634 GiB (35.6%).  See
`benchmarks/reports/2026-09-09-single-step-float32-memory.md`.

## Operational notes

- Use `big_memory=false` to select the optimized chunked path.
- For matrix input, `copy_genotypes=false` can additionally avoid the initial
  `get_genotypes` copy when the caller supplies a correctly typed dense matrix
  and transfers mutation ownership to JWAS.
- Caller-held references to the original matrix still prevent its collection.
  After building the genotype object, callers should clear their original
  matrix variable if it is no longer needed.
- Julia may retain freed pages in its allocator for reuse, so process RSS does
  not necessarily fall immediately even though the obsolete matrix is no
  longer live.
- The final phenotype-aligned dense genotype matrix remains necessary for the
  current dense MCMC sampler.  Removing that matrix requires a separate
  disk-backed or streaming single-step design.
