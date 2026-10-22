# Shared-matrix production-path benchmark

Run on 2026-10-21 with Julia 1.12.5, macOS arm64 Apple M2 Pro, one Julia thread
and one BLAS thread. Source base is feature/multiclass-fast-blocks at
7e97875ed2bb842d78465cd21e2778aa817d5293 plus the shared-alignment revision.

Both paths fit the same FARM fixed effect, LITTER random effect and genomic
effect using the actual runMCMC implementation. There were 1200 selected
animals, 900 observed training responses, 300 held-out animals and 5001 SNPs.
The benchmark used seeds 73/123, BayesC/BayesR, ordinary/50-SNP-block updates,
80 outer iterations, 20 burn-in and output frequency 5. Initialization, marker
order, phenotype order, priors and native sampler schedule match within pairs.

`metrics.csv` contains all 16 runs. Each of eight pair comparisons has animal
and marker tables joined by explicit keys, with unique/count/set checks. All
1200 expected animal keys and all 5001 expected SNP keys were verified. The
maximum absolute differences in EBV, PEV and marker-effect means were zero.

Retained genotype payload:

| Mode | Dense parents | Bytes | MiB |
|---|---:|---:|---:|
| Copying alignment | 2 | 42,008,400 | 40.06 |
| Shared row/block views | 1 | 24,004,800 | 22.89 |

The base branch already uses block views. This comparison therefore measures
removing its remaining training alignment copy, not removing a copied-block
implementation from a different branch.

`allocated_bytes` records cumulative allocations during runMCMC, not peak RSS.
Runtime includes setup, sampling and native output. On this small benchmark,
ordinary shared updates were faster, whereas fast-block shared updates were
approximately 8-15% slower than copying alignment. Strided views trade memory
for a possible performance difference; full PIC runtime must be benchmarked.
These short chains test implementation agreement, not convergence or accuracy.

Reproduce from the package root:

```bash
julia --project=. --startup-file=no benchmarks/pic_shared_matrix.jl /path/to/new_report_directory
```

Native per-run sample folders were retained locally during verification but
are omitted from the portable archive; metrics and joined comparison tables
are included. No Farm analysis or full PIC memory measurement was performed.
