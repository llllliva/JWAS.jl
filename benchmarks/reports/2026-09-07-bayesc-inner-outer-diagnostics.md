# BayesC Inner/Outer Diagnostics Benchmark

Date: 2026-09-07

## Purpose

Measure the runtime and persistent-memory cost of accumulating inner-update PIP
and marker-effect summaries for single-trait dense BayesC. The benchmark uses
the production `runMCMC` path with explicit fast-block starts and compares the
same model with `inner_outer_diagnostics` disabled and enabled.

## Configuration

- Julia 1.10.2, one thread, Apple M2 Pro
- 100 observations and 5,000 dense markers
- 300 outer iterations, 50 burn-in iterations
- Block size 10, giving 2,500 retained inner draws per marker
- Three paired random seeds: 20260907, 20260908, and 20260909

## Results

| Seed | Disabled (s) | Enabled (s) | Paired overhead |
|---:|---:|---:|---:|
| 20260907 | 0.7339 | 0.7953 | 8.4% |
| 20260908 | 0.7145 | 0.8166 | 14.3% |
| 20260909 | 0.7421 | 0.8731 | 17.7% |
| Median | 0.7339 | 0.8166 | 14.3% |

These short local runs indicate modest computational overhead, but production
Farm jobs should be measured separately because their genotype dimensions,
filesystem, and runtime are substantially different.

The accumulator stores two `UInt64` state counts, two `Float64` effect moments,
and one `UInt64` draw count per marker: 40 bytes per marker. It used 200,000
bytes for this 5,000-marker benchmark. At 1,385,033 markers, the same state is
55,401,320 bytes (55.4 MB decimal, 52.8 MiB), independent of chain length. No
per-iteration inner history is retained.

## Reproduction

Run from the repository root:

```bash
julia --project=. --startup-file=no benchmarks/bayesc_inner_outer_diagnostics.jl
```
