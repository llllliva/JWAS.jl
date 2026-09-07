# Annotation Enrichment Overhead Benchmark

**Date:** 2026-08-19

## Purpose

Measure the runtime added by `output_annotation_enrichment=true` on the
production annotated BayesC fast-block `runMCMC` path.

## Method

The reproducible driver is `benchmarks/annotation_enrichment_overhead.jl`. It
uses the same generated genotype, phenotype, and binary annotation inputs for
both modes and checks that enabled runs return all 800 expected post-burn-in
outer-iteration attempts.

| Setting | Value |
| --- | ---: |
| Observations | 100 |
| Markers | 500 |
| Binary annotation groups | 10 |
| Fast blocks | 5 |
| Chain length | 1,000 |
| Burn-in | 200 |
| Retained outer iterations | 800 |
| Replicates / seeds | 3 |
| Julia threads | 1 |
| BLAS threads | 1 |
| Julia version | 1.11.7 |

Both modes were warmed up before timing, and the enabled/disabled execution
order was alternated between replicates.

## Results

| Seed | Disabled (s) | Enabled (s) | Paired overhead |
| ---: | ---: | ---: | ---: |
| 20260819 | 2.9810 | 2.9908 | +0.33% |
| 20260820 | 2.6533 | 2.6513 | -0.07% |
| 20260821 | 2.8145 | 2.8261 | +0.41% |

Median runtime was 2.8145 seconds disabled and 2.8261 seconds enabled, a 0.41%
difference between medians. The median of the three paired overhead estimates
was 0.33%.

The raw result table is
`benchmarks/reports/2026-08-19-annotation-enrichment-overhead-results.csv`.

## Interpretation

No material overhead was detected in this local case; the measured difference
was below one percent and close to ordinary run-to-run variation. The added work
scales with markers, annotation columns, traits, and retained main/outer
iterations, so this smoke result should not be extrapolated to very large
annotation matrices without a target-scale benchmark.
