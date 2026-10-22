# PIC shared genotype matrix: implementation and verification

## Provenance

Downloaded llllliva/JWAS.jl and inspected all advertised branch tips. The newest
commit-dated branch was feature/multiclass-fast-blocks at
7e97875ed2bb842d78465cd21e2778aa817d5293 (2026-09-18). Changes are isolated in
JWAS_shared_matrix_revised_20261021, local branch pic-shared-matrix. The existing
JWAS.jl and JWAS_shared_matrix directories were preserved.

## Implementation

- Added runMCMC(...; shared_genotypes=true), default false. The requested
  repository already supported initial no-copy import and fast-block views.
- Shared alignment resolves explicit IDs to contiguous UnitRanges and stores
  row views rather than Z*X training/prediction allocations.
- The initial supported scope is one dense single-trait BayesC/BayesR term,
  including annotations and fixed/non-marker random terms, with matching
  Float32/Float64 import and MCMC precision and unit residual weights.
- Explicit guards reject duplicated/noncontiguous/permuted/repeated selections,
  incompatible precision, single-step, RRM and unsupported methods. Legacy
  alignment and the existing shared-multiclass path retain default behavior.
- Added genotype_storage_report, counting unique underlying dense parent
  matrices by object identity, not overlapping address ranges. Training and
  prediction views can be disjoint and still share the same parent.
- The shared-mode marker guard accounts for the full retained parent. Weighted
  genotype copies are prevented by the unit-weight guard.
- The paired loader adds train-first decoding before matrix allocation and
  assigns newids after ordering. Small ID/source-row masks retain original
  identities. The direct runner adopts Float32 input, never writes a genotype
  matrix file and asserts one backing matrix after production MCMC.
- The sampling schedule, priors, allele coding and centering definition are
  unchanged relative to the selected feature-branch base.

## Verification on 2026-10-21

Environment: macOS arm64, Apple M2 Pro, Julia 1.12.5; no Farm workloads launched.

- Default full package suite: **1776/1776 checks passed**. This includes
  **412 shared-matrix checks** (16 selection/layout, 386 production-MCMC
  comparisons, 10 unsupported-setting checks). The optional simulated-omics
  integration suite controlled by RUN_INTEGRATION_TESTS was not enabled.
- Direct PLINK workflow: **145/145 checks passed**, including train-first
  decoding and ten production runs covering BFP/LDP/GLP, both FIP time models,
  and BayesC/BayesR. Original BED bytes were unchanged; test responses stayed
  masked; all selected animals had finite EBV/PEV; no dense genotype file was
  saved; every run retained one Float32 backing matrix.
- A fresh CLI process using the dedicated runner Project/Manifest also passed
  the FIP-separate BayesR fast-block workflow and its post-fit one-parent check.
- Documentation build: passed. Documenter skipped deployment outside CI; no
  documentation was published.
- git diff --check: passed.

Small production tests compared copying alignment against shared alignment
using three seeds, ordinary/fast BayesC/BayesR, annotated variants, Float64 and
test-only prediction ranges. Larger production benchmarks used 1200 selected
animals, 900 training animals, 5001 SNPs, two seeds, ordinary/50-SNP blocks,
80 outer iterations, 20 burn-in, frequency 5. Every numerical comparison joined
explicit animal/marker keys and checked counts and uniqueness. The eight larger
comparisons contained all 1200 expected animal keys and all 5001 marker keys.
Maximum differences in EBV, PEV and marker posterior means were all zero.

Benchmark metrics and comparison-ready joined tables are stored under
benchmarks/reports/2026-10-21-pic-shared-matrix/. Genotype backing storage fell
from 42008400 bytes (40.06 MiB, two parents) to 24004800 bytes (22.89 MiB, one
parent). Timing and cumulative allocation were recorded from actual runMCMC,
not helper-only prototypes. Cumulative allocated bytes are not peak RSS.

## Limits

Full PIC RAM/runtime, 200k-chain convergence and prediction accuracy are not
established by synthetic tests. Small fast-block timings were approximately
8-15% slower with strided shared rows; ordinary updates were faster in this
benchmark. No production speedup is promised. Measure full-size peak RSS and
runtime on a scheduled compute-node pilot. The release is dense in-memory, not
an out-of-core sampler. Additional marker annotations and other model state
still require memory beyond the single genotype payload.
