"""
Helpers for marker-annotation setup.

Annotation handling in JWAS intentionally has two phases:

1. `get_genotypes` validates and stores the raw annotation design information and
   the user-supplied starting `Pi` without assuming how many traits the genotype
   term will affect.
2. `build_model` finalizes any annotation state that depends on the model, such
   as the number of traits and the joint inclusion states used by multi-trait
   BayesC.

This keeps the user-facing API consistent across single-trait and multi-trait
annotated methods: the same annotation matrix and starting `Pi` always go into
`get_genotypes`, while only model-dependent internal state is deferred.
"""

const ANNOTATED_BAYESC_MT_STATES = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))

# Multi-trait BayesR keeps the single-trait BayesR class convention
# internally: class 1 means zero effect, while classes 2, 3, and 4 are
# the nonzero BayesR magnitude classes.
# This is the BayesR analogue of BayesC's 0/1 inclusion indicator:
# BayesC 0 maps to BayesR class 1, and BayesC 1 maps to BayesR
# classes 2, 3, or 4 depending on the nonzero effect magnitude.
# Therefore (1, 1) is 00, (2:4, 1) is trait-1-only,
# (1, 2:4) is trait-2-only, and (2:4, 2:4) is shared.
const ANNOTATED_BAYESR_MT_STATES = let
    states = Vector{Tuple{Int,Int}}()
    push!(states, (1, 1))
    for k in 2:4
        push!(states, (k, 1))
    end
    for k in 2:4
        push!(states, (1, k))
    end
    for k in 2:4, l in 2:4
        push!(states, (k, l))
    end
    Tuple(states)
end

"""
    finalize_marker_annotation_setup!(genotypei)

Finalize any annotation state that depends on the model rather than the raw
genotype input.

`get_genotypes` handles annotation validation and stores the raw design
information. This finalizer is called from `build_model` after `genotypei.ntraits`
is known. At that point JWAS can choose the method- and trait-specific
annotation structure:

- single-trait BayesC uses one binary inclusion step
- multi-trait BayesC uses a 3-step tree over 00, 10, 01, and 11 states
- single-trait BayesR uses three ordered class-transition steps initialized from
  its requested four-class `Pi`
- multi-trait BayesR uses a 7-step tree over active patterns and BayesR
  magnitude classes
"""
function finalize_marker_annotation_setup!(genotypei::Genotypes)
    if genotypei.annotations === false
        return nothing
    end

    if genotypei.method == "BayesC"
        if genotypei.ntraits == 1
            initialize_bayesc_single_trait_annotations!(genotypei)
        elseif genotypei.ntraits > 1
            initialize_bayesc_mt_annotations!(genotypei)
        end
    elseif genotypei.method == "BayesR"
        if genotypei.ntraits == 1
            initialize_bayesr_single_trait_annotations!(genotypei)
        elseif genotypei.ntraits > 1
            initialize_bayesr_mt_annotations!(genotypei)
        end
    end
    return nothing
end

function initialize_bayesr_single_trait_annotations!(genotypei::Genotypes)
    if genotypei.annotations === false || genotypei.method != "BayesR" || genotypei.ntraits != 1
        return nothing
    end

    start_pi = bayesr_default_pi(genotypei.annotation_start_pi)
    initialize_bayesr_single_trait_annotation_coefficients!(genotypei.annotations, start_pi)

    # Keep an immutable-by-convention copy of iteration-0 Pi for diagnostics and
    # future initialization experiments. Mi.π is updated during MCMC.
    genotypei.annotation_start_pi = copy(start_pi)
    genotypei.π = copy(start_pi)
    return nothing
end

function initialize_bayesc_single_trait_annotations!(genotypei::Genotypes)
    if genotypei.annotations === false || genotypei.method != "BayesC" || genotypei.ntraits != 1
        return nothing
    end

    raw_pi = genotypei.annotation_start_pi
    raw_pi isa AbstractDict && error("Annotated BayesC genotypes initialized with a joint Pi dictionary cannot be rebuilt for a single-trait model. Use a fresh get_genotypes call with scalar/vector Pi for single-trait analysis.")

    start_pi = if raw_pi isa AbstractVector
        length(raw_pi) == genotypei.nMarkers || error("Annotated BayesC starting Pi vector length $(length(raw_pi)) must match the number of markers ($(genotypei.nMarkers)).")
        collect(Float64, raw_pi)
    else
        fill(Float64(raw_pi), genotypei.nMarkers)
    end

    design_matrix = genotypei.annotations.design_matrix
    genotypei.annotations = MarkerAnnotations(design_matrix)
    initialize_bayesc_single_trait_annotation_coefficients!(genotypei.annotations, start_pi)
    genotypei.π = start_pi
    return nothing
end

function initialize_bayesc_mt_annotations!(genotypei::Genotypes)
    if genotypei.annotations === false || genotypei.method != "BayesC" || genotypei.ntraits == 1
        return nothing
    end
    genotypei.ntraits == 2 || error("Annotated multi-trait BayesC currently supports exactly 2 traits.")

    raw_pi = genotypei.annotation_start_pi
    start_row = if raw_pi isa AbstractDict
        annotated_bayesc_mt_row_from_dict(raw_pi)
    elseif raw_pi == 0.0
        # Preserve the legacy multi-trait BayesC default: before seeing data,
        # all markers start in the all-traits-active state when Pi is omitted.
        annotated_bayesc_mt_default_row()
    elseif raw_pi isa AbstractVector{<:Real} &&
           length(raw_pi) == genotypei.nMarkers &&
           all(iszero, Float64.(raw_pi))
        annotated_bayesc_mt_default_row()
    else
        error("Annotated multi-trait BayesC requires Pi=0.0 or a joint Pi dictionary.")
    end
    validate_bayesc_mt_start_row(start_row)

    design_matrix = genotypei.annotations.design_matrix
    coeffs = zeros(Float64, size(design_matrix, 2), 3)
    genotypei.annotations = MarkerAnnotations(
        design_matrix;
        nsteps=3,
        nclasses=4,
        coefficients=coeffs,
        snp_pi=repeat(reshape(start_row, 1, :), genotypei.nMarkers, 1),
    )
    genotypei.π = annotated_bayesc_mt_pi_dict(start_row)
    return nothing
end

function initialize_bayesr_mt_annotations!(genotypei::Genotypes)
    if genotypei.annotations === false || genotypei.method != "BayesR" || genotypei.ntraits == 1
        return nothing
    end
    genotypei.ntraits == 2 || error("Annotated multi-trait BayesR currently supports exactly 2 traits.")

    start_row = annotated_bayesr_mt_default_row()
    design_matrix = genotypei.annotations.design_matrix
    coeffs = zeros(Float64, size(design_matrix, 2), 7)
    genotypei.annotations = MarkerAnnotations(
        design_matrix;
        nsteps=7,
        nclasses=length(ANNOTATED_BAYESR_MT_STATES),
        coefficients=coeffs,
        snp_pi=repeat(reshape(start_row, 1, :), genotypei.nMarkers, 1),
    )
    genotypei.π = copy(start_row)
    return nothing
end

# Default startup row for annotated 2-trait BayesC, in the
# ANNOTATED_BAYESC_MT_STATES order: 00, 10, 01, 11. This keeps the
# legacy multi-trait BayesC default that all markers start in state 11
# when Pi is omitted.
annotated_bayesc_mt_default_row() = Float64[0.0, 0.0, 0.0, 1.0]

function annotated_bayesc_mt_state_keys()
    return [Float64[state...] for state in ANNOTATED_BAYESC_MT_STATES]
end

function annotated_bayesr_mt_state_keys()
    return [Int[state...] for state in ANNOTATED_BAYESR_MT_STATES]
end

annotated_bayesr_mt_default_row() = fill(1.0 / length(ANNOTATED_BAYESR_MT_STATES), length(ANNOTATED_BAYESR_MT_STATES))

function annotated_bayesc_mt_state_index(state::AbstractVector{<:Real})
    length(state) == 2 || error("Annotated multi-trait BayesC v1 expects 2-trait state labels.")
    d1 = Int(round(Float64(state[1])))
    d2 = Int(round(Float64(state[2])))
    if d1 == 0 && d2 == 0
        return 1
    elseif d1 == 1 && d2 == 0
        return 2
    elseif d1 == 0 && d2 == 1
        return 3
    elseif d1 == 1 && d2 == 1
        return 4
    end
    error("Annotated multi-trait BayesC v1 expects binary 2-trait state labels.")
end

function annotated_bayesr_mt_state_index(state::AbstractVector{<:Integer})
    length(state) == 2 || error("Annotated multi-trait BayesR expects 2-trait class labels.")
    key = (Int(state[1]), Int(state[2]))
    for (idx, candidate) in enumerate(ANNOTATED_BAYESR_MT_STATES)
        key == candidate && return idx
    end
    error("Annotated multi-trait BayesR expects JWAS BayesR class labels in 1:4.")
end

"""
    annotated_bayesr_mt_pattern(state)

Map a 2-trait BayesR class state back to the BayesC-style active pattern:
class 1 becomes BayesC-style 0, and classes 2:4 become BayesC-style 1.
"""
function annotated_bayesr_mt_pattern(state::AbstractVector{<:Integer})
    length(state) == 2 || error("Annotated multi-trait BayesR expects 2-trait class labels.")
    return [state[1] > 1 ? 1.0 : 0.0, state[2] > 1 ? 1.0 : 0.0]
end

function annotated_bayesc_mt_row_from_dict(pi::AbstractDict)
    row = zeros(Float64, 4)
    for (state, prob) in pi
        row[annotated_bayesc_mt_state_index(state)] = Float64(prob)
    end
    isapprox(sum(row), 1.0; atol=1e-8) || error("Summation of probabilities of Pi is not equal to one.")
    return row
end

function annotated_bayesc_mt_pi_dict(row::AbstractVector{<:Real})
    length(row) == 4 || error("Annotated multi-trait BayesC v1 expects four joint prior probabilities.")
    probs = Float64.(row)
    keys = annotated_bayesc_mt_state_keys()
    return Dict(keys[i] => probs[i] for i in eachindex(keys))
end

function validate_bayesc_mt_start_row(start_row::AbstractVector{<:Real})
    length(start_row) == 4 || error("Annotated multi-trait BayesC v1 expects four joint prior probabilities.")
    trait1_mass = Float64(start_row[2]) + Float64(start_row[4])
    trait2_mass = Float64(start_row[3]) + Float64(start_row[4])
    shared_mass = Float64(start_row[4])

    trait1_mass > 0.0 || error("Annotated multi-trait BayesC requires positive startup prior mass in states {10,11} for trait 1.")
    trait2_mass > 0.0 || error("Annotated multi-trait BayesC requires positive startup prior mass in states {01,11} for trait 2.")
    shared_mass > 0.0 || error("Annotated multi-trait BayesC requires positive startup prior mass in shared state 11.")
    return nothing
end

function initialize_bayesc_single_trait_annotation_coefficients!(ann::MarkerAnnotations, start_pi::AbstractVector{<:Real})
    start_inclusion = clamp(mean(1 .- Float64.(start_pi)), eps(Float64), 1 - eps(Float64))
    ann.coefficients .= 0.0
    ann.coefficients[1] = quantile(Normal(), start_inclusion)
    ann.mu .= ann.design_matrix * ann.coefficients
    return nothing
end

"""
    initialize_bayesr_single_trait_annotation_coefficients!(ann, start_pi)

Initialize the three nested single-trait BayesR annotation probit models from a
four-class starting `Pi`. All slopes start at zero; only the three intercepts are
set, and all remain free to update during MCMC.
"""
function initialize_bayesr_single_trait_annotation_coefficients!(
    ann::MarkerAnnotations,
    start_pi::AbstractVector{<:Real},
)
    length(start_pi) == 4 || error("BayesR starting Pi must have length 4.")
    pi = Float64.(start_pi)
    all(isfinite, pi) || error("BayesR starting Pi must contain only finite values.")
    all(pi .>= 0.0) || error("BayesR starting Pi entries must be nonnegative.")
    isapprox(sum(pi), 1.0; atol=1e-8) || error("BayesR starting Pi must sum to 1.")

    nonzero = pi[2] + pi[3] + pi[4]
    larger = pi[3] + pi[4]
    nonzero > 0.0 || error("BayesR starting Pi must contain nonzero probability outside Class 1.")
    larger > 0.0 || error("BayesR starting Pi must assign positive probability to Classes 3 or 4 for the nested initialization.")

    p1 = nonzero
    p2 = larger / nonzero
    p3 = pi[4] / larger
    0.0 < p1 < 1.0 || error("Annotated BayesR requires a nondegenerate zero-vs-nonzero split in Pi.")
    0.0 < p2 < 1.0 || error("Annotated BayesR requires a nondegenerate class-2 vs class-3/4 split in Pi.")
    0.0 < p3 < 1.0 || error("Annotated BayesR requires a nondegenerate class-3 vs class-4 split in Pi.")

    size(ann.coefficients, 2) == 3 || error("Single-trait annotated BayesR requires three probit coefficient columns.")
    size(ann.mu, 2) == 3 || error("Single-trait annotated BayesR requires three latent-mean columns.")
    size(ann.snp_pi) == (size(ann.design_matrix, 1), 4) ||
        error("Single-trait annotated BayesR requires one four-class prior row per marker.")

    probs = clamp.([p1, p2, p3], eps(Float64), 1.0 - eps(Float64))
    ann.coefficients .= 0.0
    for step in 1:3
        ann.coefficients[1, step] = quantile(Normal(), probs[step])
    end
    ann.mu .= ann.design_matrix * ann.coefficients
    rebuild_bayesr_nested_priors!(ann)

    initial_pi_mean = vec(mean(ann.snp_pi, dims=1))
    isapprox(initial_pi_mean, pi; atol=1e-8) || error(
        "BayesR annotation initialization does not reproduce starting Pi. " *
        "Expected=$(pi), observed=$(initial_pi_mean)",
    )

    @info "Initialized single-trait annotated BayesR priors" starting_pi=pi nested_start_probabilities=(p1, p2, p3) initial_intercepts=copy(ann.coefficients[1, :]) mean_initial_pi=initial_pi_mean
    return nothing
end
