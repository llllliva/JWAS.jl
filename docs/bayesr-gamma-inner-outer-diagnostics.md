# Configurable BayesR gamma, inner/outer PIP, and MCMC diagnostics

This branch adds optional BayesR diagnostics and user-defined mixture-class variance multipliers while preserving the existing default behavior of JWAS and the other Bayesian alphabet methods.

## Branch and scope

The implementation is based on the fresh `master` checkout of
`reworkhow/JWAS.jl`. BayesR supports configurable class behavior and BayesC
shares the opt-in inner/outer diagnostic pathway. BayesA, BayesB, BayesL,
RR-BLUP, GBLUP, and runs with diagnostics disabled retain their original API
and behavior.

## User-defined BayesR gamma

`runMCMC` accepts:

```julia
BayesR_gamma=[0.0, 0.001, 0.1, 1.0]
```

The four values are the class multipliers. Class 1 is the null class, so its multiplier must be zero. For a shared BayesR marker variance `G`, the class variances are:

```text
class 1: 0       × G
class 2: 0.001   × G
class 3: 0.1     × G
class 4: 1.0     × G
```

The default remains:

```julia
const BAYESR_GAMMA = Float64[0.0, 0.01, 0.1, 1.0]
```

When the keyword is omitted, JWAS also accepts a comma-separated environment
value:

```bash
export BAYESR_GAMMA="0,0.001,0.1,1"
```

An explicit `BayesR_gamma=[...]` keyword takes precedence over the environment.
If `BAYESR_GAMMA` is missing or blank, JWAS uses the package default above.
Malformed or invalid environment values stop the run before MCMC starts.

Therefore, `[0.0, 0.001, 0.1, 1.0]` is a user-supplied setting and is not the default. The input is validated and copied into the model's `MCMCinfo` state. Custom gamma currently supports single-trait BayesR.

The supplied gamma is used consistently for:

1. Initial shared marker-variance calculation from the genetic variance and `Pi`.
2. Marker-class sampling.
3. The shared `G` variance update.
4. Class-variance summaries and traces.

## Inner/outer diagnostics

Enable them with:

```julia
runMCMC(model, phenotypes;
    BayesR_gamma=[0.0, 0.001, 0.1, 1.0],
    fast_blocks=block_starts,
    inner_outer_diagnostics=true)
```

This mode requires fast blocks, dense genotypes, and single-trait BayesC or
BayesR. During each post-burn-in outer iteration, the sampler records the state
and effect from each inner marker update. The common final marker-effects
columns are:

| Column | Meaning |
|---|---|
| `InnerOuter_Samples` | Number of retained inner draws for the marker |
| `InnerOuter_Model_Frequency` | Probability that the marker is included; `P(delta = 1)` for BayesC and `P(delta > 1)` for BayesR |
| `InnerOuter_Estimate`, `InnerOuter_SD` | Effect mean and SD from inner draws |

BayesR additionally reports `InnerOuter_Class1_PIP` through
`InnerOuter_Class4_PIP` and `InnerOuter_Mean_Delta`. BayesC has only excluded
and included states, so `InnerOuter_Model_Frequency` is its requested PIP.

The original outer-iteration columns remain available:

```text
Model_Frequency
Class1_Frequency
Class2_Frequency
Class3_Frequency
Class4_Frequency
MediumLarge_Frequency
```

Pi is still updated once per outer iteration from the current delta state. Inner/outer PIP is a diagnostic summary of all retained inner marker draws.

### User-controlled inner iterations

By default, the existing fast-block rule is retained: after burn-in, each block is repeated `block_size` times, while burn-in uses one repetition. To use the same number of inner iterations for every block, set:

```julia
runMCMC(model, phenotypes;
    fast_blocks=block_starts,
    inner_outer_diagnostics=true,
    BayesR_inner_iterations=3)
```

With this setting, every retained outer sweep performs exactly three inner marker-update repetitions per block, including unequal chromosome blocks. Burn-in still performs one repetition per block. The setting is BayesR fast-block specific; omitting it (`false`) preserves the previous block-size behavior.

Farm or other batch wrappers may also provide these settings through the
environment:

```bash
export BAYESR_BLOCK_STARTS="1,5001,9201"
export BAYESR_INNER_ITERATIONS="300"  # or false/auto
```

When `runMCMC` receives `fast_blocks=true`, `BAYESR_BLOCK_STARTS` is resolved
to an explicit block-start vector, so the requested chain length is interpreted
as outer sweeps and is not divided by the automatic square-root block size.
When `BayesR_inner_iterations=false`, a positive environment value supplies a
fixed count. If the environment value is absent, `false`, or `auto`, JWAS
calculates inner repetitions separately from each block's marker count. Direct
numeric/vector keyword arguments take precedence over these fallbacks.

## MCMC output files

For BayesR, the sampler writes the following additional files.

### Delta trace

```text
MCMC_samples_delta_<genotype>_<trait>.txt
```

The first row contains marker IDs. Each subsequent saved row contains integer class assignments:

- `1`: null class
- `2`: class 2
- `3`: class 3
- `4`: class 4

This is one row per saved MCMC/outer iteration. Use `output_samples_frequency=1` to save every outer iteration. In fast-block mode, the row is the final delta state after that outer iteration's inner sweep. The delta file is written even when `output_marker_effect_samples=false`.

### Class-variance trace

```text
MCMC_samples_bayesr_class_variances_<genotype>.txt
```

Each row contains the four current class variances, calculated as `BayesR_gamma .* G`.

### Class-variance summary

```text
BayesR_class_variances_<genotype>.txt
```

The summary contains `Class`, `Gamma`, `Estimate`, and `SD` columns. The null-class estimate is zero; the other estimates summarize the posterior values of `gamma[k] * G`.

## Default behavior and compatibility

When `BayesR_gamma` is omitted and `BAYESR_GAMMA` is missing or blank, the original default gamma is retained. The explicit keyword overrides the environment. Other Bayesian alphabet methods do not use the BayesR gamma state.

## Validation

The focused test covers custom gamma, runtime-option fallback, automatic and fixed inner iterations, inner/outer PIP, class-variance outputs, and delta traces. The full test suite passed with 1,016 tests.
