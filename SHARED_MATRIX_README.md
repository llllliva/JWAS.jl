# PIC JWAS: one shared Float32 genotype matrix

This revision adopts one selected train+test genotype matrix, fits through
contiguous training-row views, and uses fast-block views of the same allocation.
Posterior genomic EBVs and PEVs remain available for all selected animals.
The direct PLINK workflow does not save a dense genotype matrix.

Source: [llllliva/JWAS.jl](https://github.com/llllliva/JWAS.jl), newest advertised
feature branch by commit date at download: `feature/multiclass-fast-blocks`,
base commit `7e97875ed2bb842d78465cd21e2778aa817d5293`. The base already used
block views; this revision removes plain-Matrix training alignment copies too.
Existing local JWAS checkouts were preserved. Changes are local, not pushed.

## Install the two environments

Keep `JWAS_shared_matrix_revised_20261021/` and
`JWAS_raw_phenotype_models/` beside each other, as in the release archive.
From their parent directory:

```bash
julia --project=JWAS_shared_matrix_revised_20261021 -e 'using Pkg; Pkg.instantiate()'
julia --project=JWAS_raw_phenotype_models -e 'using Pkg; Pkg.instantiate()'
```

Tested with Julia 1.12.5. Run real PIC analyses in scheduled compute jobs with
sufficient memory; the commands below show the application command only.

## Start directly from bfiles

Substitute actual phenotype, split and PLINK paths. `--bfile` omits the file
extension and original IDENTs must match FAM IID by default. `--fam-id fid`
or an explicit `--plink-id-map` can be supplied if that is not the ID convention.

```bash
julia --project=JWAS_raw_phenotype_models \
  JWAS_raw_phenotype_models/run_pic_jwas_from_bfile.jl \
  --bfile /path/to/LG02 \
  --trait FIP --pheno /path/to/LG02_FIP.txt \
  --train-ids /path/to/LG02.FIP.ALS.train.txt \
  --test-ids /path/to/LG02.FIP.ALS.test.txt \
  --event-model separate --method BayesC --pi 0.99 \
  --fast-blocks 100 --chain 200000 --burnin 400 --output-frequency 5 \
  --outdir results/LG02_FIP_ALS_shared
```

The main runner defaults to the paired JWAS path, train-first decoding and
shared alignment. The prepared-input runner and preparation script use matching
defaults too. The older shared-wrapper name remains an alias. With omitted
`--pi`, BayesC uses 0.99 and BayesR uses 0.95,0.02,0.02,0.01. The revised API
is checked before decoding the large matrix. Newids are assigned after train-first ordering. All
phenotype rows, source masks, FID/IID maps and split lists follow those newids.
Do not mix earlier source-order newids with these outputs. Training-only missing
genotype means are used when decoding BED. Test responses are masked before
fitting. WSP/WEP remain excluded, and the agreed model effects are unchanged.

For the second FIP model use `--event-model combined` and a different output
directory. For BFP/LDP/GLP change the trait/phenotype/split inputs; combined BYRMO
is automatic. For BayesR use `--method BayesR --pi 0.95,0.02,0.02,0.01`.
Use `--fast-blocks false` for ordinary updates. Use one fresh process per model.

Numeric fast blocks still divide nominal chain length by block size. The
example runs 2000 outer iterations, discards 400 outer iterations and saves 320
draws. It is a pilot schedule, not a convergence guarantee. Ordinary updates
instead run all 200000 outer iterations; reconsider burn-in and runtime when
switching samplers.

## Memory verification and sizing

Every shared run saves `genotype_storage_report.csv` and asserts:

- Exactly one dense genotype backing matrix.
- Training and output share the same parent allocation.
- All fast blocks share the training parent.
- The registered genotype type is Float32.

For A selected animals, P SNPs and block size B, main Float32 payload is about
`4*P*A + 4*P*B` bytes; the second term is the block cross-product cache. Model,
marker, annotation and Julia state still add RAM.

At 1.1M SNPs and block size 100, the conservative provisional per-job requests
below include an 8 GiB allowance, 30% headroom and upward 32G rounding. They
use local audited animal counts, before fresh FAM matching. Here ALS and 5LS
have identical train+test totals, so requests match between splits.

| Line | BFP | LDP | GLP | FIP |
|---|---:|---:|---:|---:|
| LG02 | 384G | 384G | 384G | 64G |
| LG03 | 352G | 352G | 352G | 64G |
| LG15 | 352G | 352G | 384G | 96G |
| LG27 | 224G | 224G | 224G | 96G |
| LG62 | 160G | 192G | 192G | 64G |
| LG65 | 512G | 544G | 544G | 128G |

These are total per-job Slurm G units, not per CPU, and remain theoretical
requests until a full-size pilot measures MaxRSS. The CSV in the runner folder
contains all 48 ALS/5LS count/calculation rows. Each FIP variant needs its own
allocation if run simultaneously. No cluster partition/account was selected.

## Verification and scope

All 1776 default package checks, 145 direct-PLINK checks and the documentation
build passed. A fresh standalone CLI run also passed. Production MCMC tests
covered ordinary/fast BayesC/BayesR, annotations, several seeds, Float32/Float64
and test-only prediction subsets. Eight larger synthetic comparisons joined
all expected animal/SNP keys: EBV, PEV and marker-effect differences were zero.
Retained genotype backing storage fell from 40.06 MiB to 22.89 MiB in the
1200-animal, 5001-SNP benchmark. Timings are benchmarks, not production guarantees.

The initial shared API supports one dense genotype term and one trait, with
contiguous unique training/prediction selections and unit residual weights.
Repeated records, arbitrary ID permutations, single-step and RRM are rejected.
The four PIC input traits use one phenotype row per animal in this workflow.
Full PIC peak RAM, runtime, convergence and prediction accuracy remain unmeasured.

Core API details: `docs/src/manual/pic_shared_matrix.md`.
Implementation/test record: `docs/plans/2026-10-21-pic-shared-matrix-implementation.md`.
Benchmark data: `benchmarks/reports/2026-10-21-pic-shared-matrix/metrics.csv`.
The default alignment remains available with `shared_genotypes=false`.
