# Shared Dense Multiclass BayesC

`get_shared_genotype_classes` creates overlapping BayesC genotype classes
without copying their genotype columns. Each returned class is a normal JWAS
genotype term, but its matrix is a column view into one shared dense matrix.

```julia
using JWAS

# Load and preprocess the complete marker matrix once. For an already-QCed
# Float32 matrix, these options avoid an initial input copy as well.
base = get_genotypes(
    genotype_matrix,
    1.0;
    method="BayesC",
    Pi=0.99,
    quality_control=false,
    copy_genotypes=false,
)
base.obsID .= genotype_ids
base.markerID .= marker_ids

# Rows follow base.markerID; columns are annotation classes. Overlap is valid.
classes = get_shared_genotype_classes(
    base,
    membership_matrix;
    add_unassigned=true,
    class_names=["Coding", "Enhancer", "Others"],
    Pi=0.99,
)

# JWAS model terms are resolved from top-level variable names.
global coding = classes[1]
global enhancer = classes[2]
global others = classes[3]
model = build_model("y = intercept + coding + enhancer + others", 1.0)
```

The membership matrix must be binary and must follow the source object's
post-QC marker order. Create the shared classes only after all imputation,
centering, QC, and ID assignment are complete. Include the returned classes in
the model, not `base` itself.

Every class has separate marker effects, inclusion indicators, pi, and
marker-effect variance. An overlapping SNP therefore retains the ordinary
multiclass statistical interpretation while its physical genotype column is
stored once. This reduces genotype storage but does not make overlapping
class-specific effects identifiable from the likelihood.

## Class-Specific Heritability

For a single-trait dense multiclass BayesC model, JWAS automatically writes

```text
MCMC_samples_class_heritability.txt
```

when `output_heritability=true` (the default). Columns are the genotype-term
names (`coding`, `enhancer`, and `others` in the example), and each subsequent
row is one retained MCMC iteration according to `burnin` and
`output_samples_frequency`. This file does not require `outputEBV=true`.

For class genetic value `g_c` and total marker genetic value
`g = sum_c g_c`, the reported contribution is

```math
h_c^2 = \frac{Cov(g_c,g)}{Var(g)+\sigma_e^2}.
```

This allocation includes cross-class covariance exactly once, so the class
columns sum to total marker heritability. A class contribution can be negative
when its genetic value is negatively correlated with the total; JWAS does not
truncate it. Using `Var(g_c)` instead would not conserve total heritability for
correlated or overlapping classes.

For single-trait Annotated BayesC, the marker-specific pi sample file is also
row-oriented in this version. Its first row contains marker IDs; every later
row is one retained MCMC iteration and every column is one marker.

## Fast blocks

Shared multiclass models support `fast_blocks=true` or a numeric block size:

```julia
out = runMCMC(model, phenotypes;
    fast_blocks=100,
    chain_length=140_000,
    burnin=300,
    independent_blocks=false,
    inner_outer_diagnostics=true,
    output_marker_effect_samples=false,
)
```

Numeric sizes generate separate block boundaries in each class's local marker
order. Every marker is included, including short final blocks and classes with
only one marker. `true` selects `floor(sqrt(nObs))` as the requested block size.
An explicit vector such as `[1, 3]` is applied in local coordinates to every
class and must be valid for all classes; it is not a list of source SNP indices.
For unequal classes, prefer a numeric size.

Classes and their blocks update sequentially using the same residual when
`independent_blocks=false`. Overlapping memberships keep distinct effects and
indicators. Genotype blocks remain views; block crossproducts require extra
memory approximately proportional to the total class memberships times block
size for classes much larger than a block.

The existing iteration convention is retained: numeric/true settings run
`floor(chain_length / requested_block_size)` outer iterations. The example runs
1,400 outer iterations with 300 outer burn-in iterations. Burn-in and output
frequency are measured in outer iterations. BayesC performs as many inner
repetitions as the actual width of each block, including during burn-in; short
blocks therefore receive fewer repetitions. Explicit start vectors do not
rescale the outer chain length. Choose a chain length that leaves retained
outer iterations, and assess mixing for class effects and hyperparameters.

Inner/outer diagnostics are accumulated separately for each class. The usual
outer PIP and class heritability outputs remain available. This does not imply
a measured speedup: benchmark time and effective samples on your data before
choosing a production schedule.
