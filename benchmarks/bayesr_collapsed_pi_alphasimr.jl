#!/usr/bin/env julia

using CSV
using DataFrames
using DelimitedFiles
using LinearAlgebra
using Statistics
using JWAS

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_ROOT = normpath(joinpath(
    PROJECT_ROOT,
    "..",
    "Test_samples",
    "JWASvsBayesR3",
    "AlphaSimR_smallNe_architecture",
))
const DEFAULT_OUT_ROOT = joinpath(DATA_ROOT, "jwas_collapsed_pi_comparison")

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_int_list(name, default) = parse.(Int, split(get(ENV, name, default), ','))
env_float_list(name, default) = parse.(Float64, split(get(ENV, name, default), ','))

const CHAIN_LENGTH = env_int("BAYESR_COLLAPSED_CHAIN", 300)
const BURNIN = env_int("BAYESR_COLLAPSED_BURNIN", 30)
const OUTPUT_FREQUENCY = env_int("BAYESR_COLLAPSED_OUTPUT_FREQUENCY", 5)
const SEEDS = env_int_list("BAYESR_COLLAPSED_SEEDS", "20260803,20260804")
const GAMMA = env_float_list("BAYESR_COLLAPSED_GAMMA", "0.0,0.1,1.0,10.0")
const STARTING_PI = env_float_list("BAYESR_COLLAPSED_STARTING_PI", "0.94,0.049,0.01,0.001")
const OUT_ROOT = abspath(get(ENV, "BAYESR_COLLAPSED_OUT_ROOT", DEFAULT_OUT_ROOT))
const ARCHITECTURES = ["small_effect_polygenic", "sparse_large_effect"]
const SAMPLERS = [:standard, :collapsed]

CHAIN_LENGTH > BURNIN || error("BAYESR_COLLAPSED_CHAIN must exceed BAYESR_COLLAPSED_BURNIN.")
OUTPUT_FREQUENCY > 0 || error("BAYESR_COLLAPSED_OUTPUT_FREQUENCY must be positive.")
length(GAMMA) == 4 || error("BAYESR_COLLAPSED_GAMMA must contain four values.")
length(STARTING_PI) == 4 || error("BAYESR_COLLAPSED_STARTING_PI must contain four values.")
isapprox(sum(STARTING_PI), 1.0; atol=1e-8) || error("Starting Pi must sum to one.")
isdir(OUT_ROOT) && error("Output root already exists: $OUT_ROOT")
mkpath(OUT_ROOT)

function require_unique(values, label)
    length(values) == length(unique(values)) || error("Duplicate $label values were found.")
end

function align_truth(marker_output, architecture)
    truth_path = joinpath(DATA_ROOT, architecture, "truth.csv")
    truth = CSV.read(truth_path, DataFrame)
    truth.Marker = string.(truth.Marker)
    markers = string.(marker_output.Marker_ID)
    require_unique(markers, "marker-output Marker_ID")
    require_unique(truth.Marker, "truth Marker")
    length(markers) == nrow(truth) ||
        error("Marker count mismatch: output=$(length(markers)), truth=$(nrow(truth)).")
    Set(markers) == Set(truth.Marker) || error("Marker ID sets do not match the truth table.")

    estimates = DataFrame(
        Marker=markers,
        Estimate=Float64.(marker_output.Estimate),
        PIP=Float64.(marker_output.Model_Frequency),
        PIP_class1=Float64.(marker_output.Class1_Frequency),
        PIP_class2=Float64.(marker_output.Class2_Frequency),
        PIP_class3=Float64.(marker_output.Class3_Frequency),
        PIP_class4=Float64.(marker_output.Class4_Frequency),
    )
    aligned = innerjoin(estimates, truth, on=:Marker, validate=(true, true))
    nrow(aligned) == nrow(truth) || error("Truth join dropped markers.")
    return aligned
end

function prediction_metrics(aligned, architecture, intercept)
    genotype_path = joinpath(DATA_ROOT, "genotypes_test.csv")
    phenotype_path = joinpath(DATA_ROOT, architecture, "phenotypes_test.csv")
    genotypes = CSV.read(genotype_path, DataFrame)
    phenotypes = CSV.read(phenotype_path, DataFrame)
    genotype_ids = string.(genotypes.ID)
    phenotype_ids = string.(phenotypes.ID)
    require_unique(genotype_ids, "test genotype ID")
    require_unique(phenotype_ids, "test phenotype ID")
    Set(genotype_ids) == Set(phenotype_ids) || error("Test genotype and phenotype ID sets differ.")

    phenotype_lookup = Dict(string(row.ID) => (Float64(row.y), Float64(row.genetic_value))
                            for row in eachrow(phenotypes))
    marker_symbols = Symbol.(aligned.Marker)
    missing_markers = setdiff(marker_symbols, propertynames(genotypes))
    isempty(missing_markers) || error("Test genotypes are missing $(length(missing_markers)) markers.")
    design = Matrix{Float64}(select(genotypes, marker_symbols))
    prediction = intercept .+ design * aligned.Estimate
    observed = [phenotype_lookup[id][1] for id in genotype_ids]
    true_genetic_value = [phenotype_lookup[id][2] for id in genotype_ids]
    return (
        phenotype_correlation=cor(prediction, observed),
        genetic_value_correlation=cor(prediction, true_genetic_value),
        prediction_rmse=sqrt(mean((prediction .- observed) .^ 2)),
    )
end

function top_causal_count(aligned, n)
    selected = partialsortperm(aligned.PIP, 1:min(n, nrow(aligned)); rev=true)
    return count(identity, Bool.(aligned.Causal[selected]))
end

function numeric_trace(path)
    values = Float64[]
    for token in split(read(path, String), r"[\s,]+")
        isempty(token) && continue
        value = tryparse(Float64, token)
        value === nothing || push!(values, value)
    end
    return values
end

function mean_trace(path)
    values = numeric_trace(path)
    isempty(values) && return NaN
    return mean(values)
end

summary_rows = NamedTuple[]
marker_tables = Dict{Tuple{String,Int,Symbol},DataFrame}()

for architecture in ARCHITECTURES
    phenotype_path = joinpath(DATA_ROOT, architecture, "phenotypes_train.csv")
    genotype_path = joinpath(DATA_ROOT, "genotypes_train.csv")
    phenotypes = select(CSV.read(phenotype_path, DataFrame), [:ID, :y])
    phenotypes.ID = string.(phenotypes.ID)

    for seed in SEEDS
        for sampler in SAMPLERS
            run_dir = joinpath(
                OUT_ROOT,
                "chain_$(CHAIN_LENGTH)_burnin_$(BURNIN)",
                architecture,
                "seed_$(seed)",
                String(sampler),
            )
            mkpath(dirname(run_dir))
            println("Running architecture=$architecture sampler=$sampler seed=$seed")
            flush(stdout)

            global geno = get_genotypes(
                genotype_path,
                1.0;
                header=true,
                separator=',',
                method="BayesR",
                Pi=STARTING_PI,
                estimatePi=true,
                G_is_marker_variance=false,
                estimate_variance=true,
                estimate_scale=false,
                quality_control=false,
                center=false,
            )
            model = build_model("y = intercept + geno", 1.0)
            output_ref = Ref{Any}()
            runtime_seconds = @elapsed begin
                output_ref[] = runMCMC(
                    model,
                    phenotypes;
                    chain_length=CHAIN_LENGTH,
                    burnin=BURNIN,
                    output_samples_frequency=OUTPUT_FREQUENCY,
                    output_folder=run_dir,
                    output_marker_effect_samples=false,
                    seed=seed,
                    printout_model_info=false,
                    printout_frequency=CHAIN_LENGTH + 1,
                    outputEBV=false,
                    output_heritability=false,
                    fast_blocks=false,
                    memory_guard=:off,
                    BayesR_gamma=GAMMA,
                    BayesR_pi_sampler=sampler,
                )
            end
            output = output_ref[]
            marker_output = output["marker effects geno"]
            aligned = align_truth(marker_output, architecture)
            aligned.Sampler = fill(String(sampler), nrow(aligned))
            aligned.Seed = fill(seed, nrow(aligned))
            marker_tables[(architecture, seed, sampler)] = aligned
            CSV.write(joinpath(run_dir, "pip_truth_aligned.csv"), aligned)

            location = output["location parameters"]
            intercept_rows = findall((location.Effect .== "intercept") .&
                                     (location.Level .== "intercept"))
            length(intercept_rows) == 1 || error("Expected exactly one intercept row.")
            intercept = Float64(location.Estimate[only(intercept_rows)])
            prediction = prediction_metrics(aligned, architecture, intercept)
            causal = Bool.(aligned.Causal)
            null = .!causal
            pi_estimate = Float64.(output["pi_geno"].Estimate)
            shared_g = mean_trace(joinpath(run_dir, "MCMC_samples_marker_effects_variances_geno.txt"))
            residual_variance = mean_trace(joinpath(run_dir, "MCMC_samples_residual_variance.txt"))
            pi_trace_values = numeric_trace(joinpath(run_dir, "MCMC_samples_pi_geno.txt"))
            length(pi_trace_values) % 4 == 0 || error("Pi trace does not contain complete four-class samples.")
            saved_samples = div(length(pi_trace_values), 4)

            row = (
                architecture=architecture,
                sampler=String(sampler),
                seed=seed,
                chain_length=CHAIN_LENGTH,
                burnin=BURNIN,
                saved_samples=saved_samples,
                runtime_seconds=runtime_seconds,
                n_markers=nrow(aligned),
                n_causal=count(causal),
                mean_pip=mean(aligned.PIP),
                mean_pip_causal=mean(aligned.PIP[causal]),
                mean_pip_null=mean(aligned.PIP[null]),
                pip_truth_correlation=cor(aligned.PIP, Float64.(causal)),
                causal_top10=top_causal_count(aligned, 10),
                causal_top20=top_causal_count(aligned, 20),
                causal_top100=top_causal_count(aligned, 100),
                pi1=pi_estimate[1],
                pi2=pi_estimate[2],
                pi3=pi_estimate[3],
                pi4=pi_estimate[4],
                shared_marker_variance=shared_g,
                residual_variance=residual_variance,
                phenotype_prediction_correlation=prediction.phenotype_correlation,
                genetic_value_prediction_correlation=prediction.genetic_value_correlation,
                phenotype_prediction_rmse=prediction.prediction_rmse,
                gamma=join(GAMMA, ';'),
                starting_pi=join(STARTING_PI, ';'),
            )
            push!(summary_rows, row)
            CSV.write(joinpath(OUT_ROOT, "run_summary.csv"), DataFrame(summary_rows))
            println("Completed architecture=$architecture sampler=$sampler seed=$seed in $(round(runtime_seconds, digits=2)) seconds")
            flush(stdout)
        end
    end
end

pairwise_rows = NamedTuple[]
for architecture in ARCHITECTURES, seed in SEEDS
    standard = marker_tables[(architecture, seed, :standard)]
    collapsed = marker_tables[(architecture, seed, :collapsed)]
    standard_for_join = select(standard, :Marker, :PIP => :PIP_standard)
    collapsed_for_join = select(collapsed, :Marker, :PIP => :PIP_collapsed)
    joined = innerjoin(standard_for_join, collapsed_for_join, on=:Marker, validate=(true, true))
    nrow(joined) == nrow(standard) == nrow(collapsed) || error("Pairwise marker join dropped rows.")
    Set(joined.Marker) == Set(standard.Marker) || error("Pairwise marker keys do not match.")
    CSV.write(joinpath(OUT_ROOT, "pip_pairwise_$(architecture)_seed_$(seed).csv"), joined)
    standard_top100 = Set(standard.Marker[partialsortperm(standard.PIP, 1:100; rev=true)])
    collapsed_top100 = Set(collapsed.Marker[partialsortperm(collapsed.PIP, 1:100; rev=true)])
    push!(pairwise_rows, (
        architecture=architecture,
        seed=seed,
        pip_correlation=cor(joined.PIP_standard, joined.PIP_collapsed),
        mean_pip_difference=mean(joined.PIP_collapsed .- joined.PIP_standard),
        mean_absolute_pip_difference=mean(abs.(joined.PIP_collapsed .- joined.PIP_standard)),
        top100_overlap=length(intersect(standard_top100, collapsed_top100)),
    ))
end
CSV.write(joinpath(OUT_ROOT, "pairwise_summary.csv"), DataFrame(pairwise_rows))

settings = DataFrame(
    setting=["chain_length", "burnin", "output_frequency", "seeds", "gamma", "starting_pi", "fast_blocks"],
    value=[
        string(CHAIN_LENGTH),
        string(BURNIN),
        string(OUTPUT_FREQUENCY),
        join(SEEDS, ';'),
        join(GAMMA, ';'),
        join(STARTING_PI, ';'),
        "false",
    ],
)
CSV.write(joinpath(OUT_ROOT, "settings.csv"), settings)
println("Wrote collapsed-pi comparison to $OUT_ROOT")
