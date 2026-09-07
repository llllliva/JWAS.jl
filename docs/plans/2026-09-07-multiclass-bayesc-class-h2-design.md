# Multiclass BayesC class heritability design

## Current behavior

JWAS writes only overall heritability based on the summed genetic value. It does
not write one heritability contribution per multiclass BayesC term.

## Definition

For class `c`, let `g_c = M_c alpha_c` and let `g = sum_c g_c`. Define the
class-attributed genetic variance as

```math
V_c = Cov(g_c, g).
```

Then the class heritability contribution is

```math
h_c^2 = V_c / (Var(g) + sigma_e^2).
```

This covariance allocation is required for overlapping classes. Because

```math
sum_c Cov(g_c,g) = Var(g),
```

the class contributions sum exactly to total marker heritability. Reporting
`Var(g_c)` instead would double-count cross-class covariance and would not
conserve total heritability. A class contribution may be negative when its
genetic value is negatively correlated with the total; this is meaningful and
must not be truncated.

## Output

For single-trait dense models with two or more BayesC genotype terms and
`output_heritability=true`, write:

```text
MCMC_samples_class_heritability.txt
```

Columns are genotype-term/class names and rows follow burn-in plus
`output_samples_frequency`. The calculation uses the aligned training genotype
matrices and does not require `outputEBV=true`.

## Memory

Use one `nobservations by nclasses` temporary matrix of class genetic values and
one total vector per retained draw. This does not copy any genotype matrix. The
temporary memory is approximately `8 * nobs * (nclasses + 1)` bytes.

## Tests

- Verify the file is created with class names as headers.
- Verify retained row count.
- Verify each row sums to independently calculated total marker heritability.
- Verify the feature does not require EBV output and preserves shared genotype
  aliases.
