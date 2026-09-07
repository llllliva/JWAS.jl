using Test
using JWAS
using DataFrames
using Distributions
using CSV

const ANNOTATION_ENRICHMENT_G = Float32[
    0 0 0 2
    0 1 1 1
    1 0 2 0
    1 2 0 1
    2 1 1 2
    2 2 2 0
]

const ANNOTATION_ENRICHMENT_A = Float64[
    1 1
    1 0
    0 1
    0 0
]

function annotation_enrichment_genotype(name::Symbol)
    genotype_df = DataFrame(ANNOTATION_ENRICHMENT_G, Symbol.("m" .* string.(1:4)))
    insertcols!(genotype_df, 1, :ID => string.(1:size(genotype_df, 1)))
    genotype = get_genotypes(
        genotype_df,
        1.0;
        method="BayesC",
        Pi=0.2,
        estimatePi=true,
        annotations=ANNOTATION_ENRICHMENT_A,
        quality_control=false,
    )
    @eval Main global $(name) = $genotype
    return genotype
end

@testset "Annotation heritability enrichment" begin
    @testset "draw formula" begin
        genotype = annotation_enrichment_genotype(:annotation_enrichment_formula_geno)
        genotype.ntraits = 1
        genotype.trait_names = ["y"]
        genotype.α = [Float32[1.0, 2.0, 0.5, 1.5]]
        genotype.annotations.enrichment_stats = JWAS.initialize_annotation_enrichment(genotype)
        genotype.annotations.coefficients .= [0.25, 1.2, -0.35]
        genotype.annotations.mu .= genotype.annotations.design_matrix * genotype.annotations.coefficients

        JWAS.update_annotation_enrichment!(genotype, 2.0)
        result = JWAS.annotation_enrichment_dataframe(genotype)

        twopq = vec(2.0 .* genotype.alleleFreq .* (1.0 .- genotype.alleleFreq))
        marker_variance = twopq .* Float64.(genotype.α[1]).^2
        eta_overlap = sum(genotype.annotations.coefficients)
        p_overlap = cdf(Normal(), eta_overlap)
        overlap_scores = [
            abs(p_overlap - cdf(Normal(), eta_overlap - genotype.annotations.coefficients[2])),
            abs(p_overlap - cdf(Normal(), eta_overlap - genotype.annotations.coefficients[3])),
        ]
        overlap_shares = overlap_scores ./ sum(overlap_scores)
        group_variance = [
            marker_variance[2] + overlap_shares[1] * marker_variance[1],
            marker_variance[3] + overlap_shares[2] * marker_variance[1],
            marker_variance[4],
        ]
        total_variance = sum(marker_variance)
        expected_proportion = group_variance ./ total_variance
        expected_h2 = group_variance ./ (total_variance + 2.0)
        expected_marker_fractions = [0.375, 0.375, 0.25]
        expected_enrichment = expected_proportion ./ expected_marker_fractions

        @test result.Heritability_Estimate ≈ expected_h2
        @test result.H2_Proportion_Estimate ≈ expected_proportion
        @test result.Enrichment_Estimate ≈ expected_enrichment
        @test sum(result.H2_Proportion_Estimate) ≈ 1.0
        @test sum(result.Heritability_Estimate) ≈ total_variance / (total_variance + 2.0)
        @test sum(result.SNP_Fraction .* result.Enrichment_Estimate) ≈ 1.0
        @test result.Posterior_Iterations == [1, 1, 1]
        @test result.Posterior_Samples == [1, 1, 1]
        @test result.Annotation == ["Annotation_1", "Annotation_2", "Unannotated"]
        @test result.SNP_Count == [2, 2, 1]
        @test result.SNP_Fraction ≈ expected_marker_fractions
        @test all(result.Allocation_Method .== "probit_leave_one_out_partition")
        @test all(result.Heritability_SD .== 0.0)
    end

    @testset "zero-score equal split" begin
        genotype = annotation_enrichment_genotype(:annotation_enrichment_equal_split_geno)
        genotype.ntraits = 1
        genotype.trait_names = ["y"]
        genotype.α = [Float32[1.0, 1.0, 1.0, 1.0]]
        genotype.annotations.enrichment_stats = JWAS.initialize_annotation_enrichment(genotype)
        genotype.annotations.coefficients .= 0.0

        marker_variance = genotype.annotations.enrichment_stats.marker_weights .* abs2.(Float64.(genotype.α[1]))
        group_variance = JWAS.bayesc_overlap_group_variance(
            genotype,
            marker_variance,
            @view(genotype.annotations.design_matrix[:, 2:end]),
        )
        @test group_variance ≈ [marker_variance[2] + marker_variance[1] / 2,
                                marker_variance[3] + marker_variance[1] / 2,
                                marker_variance[4]]
        @test sum(group_variance) ≈ sum(marker_variance)
    end

    @testset "binary annotation validation" begin
        nonbinary = copy(ANNOTATION_ENRICHMENT_A)
        nonbinary[1, 1] = 0.5
        genotype_df = DataFrame(ANNOTATION_ENRICHMENT_G, Symbol.("m" .* string.(1:4)))
        insertcols!(genotype_df, 1, :ID => string.(1:size(genotype_df, 1)))
        genotype = get_genotypes(
            genotype_df,
            1.0;
            method="BayesC",
            Pi=0.2,
            annotations=nonbinary,
            quality_control=false,
        )
        genotype.ntraits = 1
        @test_throws ErrorException JWAS.initialize_annotation_enrichment(genotype)
    end

    for (label, fast_blocks) in (("standard", false), ("fast", [1, 3]))
        @testset "$label iteration cadence" begin
            symbol = Symbol("annotation_enrichment_", label, "_geno")
            genotype = annotation_enrichment_genotype(symbol)
            model = build_model("y = intercept + $(symbol)", 1.0)
            phenotypes = DataFrame(
                ID=string.(1:6),
                y=Float64[1.2, -0.4, 0.8, -1.0, 0.5, -0.2],
            )
            output_folder = "test_annotation_enrichment_$label"
            output = runMCMC(
                model,
                phenotypes;
                chain_length=8,
                burnin=2,
                output_samples_frequency=3,
                outputEBV=false,
                output_heritability=false,
                output_marker_effect_samples=false,
                output_folder=output_folder,
                fast_blocks=fast_blocks,
                memory_guard=:off,
                printout_model_info=false,
                seed=781,
            )

            key = "annotation heritability enrichment $(symbol)"
            @test haskey(output, key)
            result = output[key]
            @test nrow(result) == 3
            @test all(result.Posterior_Iterations .== 6)
            @test all(0 .< result.Posterior_Samples .<= 6)
            @test all(isfinite, result.Heritability_Estimate)
            @test all(isfinite, result.H2_Proportion_Estimate)
            @test all(isfinite, result.Enrichment_Estimate)
            @test all(result.Heritability_Estimate .>= 0.0)
            @test sum(result.H2_Proportion_Estimate) ≈ 1.0
            @test sum(result.SNP_Fraction .* result.Enrichment_Estimate) ≈ 1.0
            @test result.Annotation[end] == "Unannotated"
            @test isfile(joinpath(output_folder, replace(key, " " => "_") * ".txt"))

            h2_file = joinpath(output_folder, "MCMC_samples_annotation_h2_$(symbol).txt")
            enrichment_file = joinpath(output_folder, "MCMC_samples_annotation_enrichment_$(symbol).txt")
            @test isfile(h2_file)
            @test isfile(enrichment_file)
            h2_samples = CSV.read(h2_file, DataFrame)
            enrichment_samples = CSV.read(enrichment_file, DataFrame)
            @test names(h2_samples) == result.Annotation
            @test names(enrichment_samples) == result.Annotation
            @test nrow(h2_samples) == 2
            @test nrow(enrichment_samples) == 2
            h2_matrix = Matrix(h2_samples)
            enrichment_matrix = Matrix(enrichment_samples)
            valid_h2 = vec(all(isfinite, h2_matrix; dims=2))
            valid_enrichment = vec(all(isfinite, enrichment_matrix; dims=2))
            @test valid_h2 == valid_enrichment
            @test any(valid_h2)
            @test all(valid_h2 .| vec(all(isnan, h2_matrix; dims=2)))
            @test all(valid_enrichment .| vec(all(isnan, enrichment_matrix; dims=2)))
            weighted_enrichment = vec(sum(enrichment_matrix[valid_enrichment, :] .* transpose(result.SNP_Fraction), dims=2))
            @test all(abs.(weighted_enrichment .- 1.0) .< 1e-10)
            h2_sums = vec(sum(h2_matrix[valid_h2, :], dims=2))
            @test all((h2_sums .> 0.0) .& (h2_sums .< 1.0))
            rm(output_folder; recursive=true)
        end
    end

    @testset "annotated BayesR fast-block output" begin
        genotype_df = DataFrame(ANNOTATION_ENRICHMENT_G, Symbol.("m" .* string.(1:4)))
        insertcols!(genotype_df, 1, :ID => string.(1:size(genotype_df, 1)))
        global annotation_enrichment_bayesr_geno = get_genotypes(
            genotype_df,
            1.0;
            method="BayesR",
            Pi=[0.1, 0.3, 0.3, 0.3],
            estimatePi=true,
            annotations=ANNOTATION_ENRICHMENT_A,
            quality_control=false,
        )
        model = build_model("y = intercept + annotation_enrichment_bayesr_geno", 1.0)
        phenotypes = DataFrame(
            ID=string.(1:6),
            y=Float64[1.2, -0.4, 0.8, -1.0, 0.5, -0.2],
        )
        output_folder = "test_annotation_enrichment_bayesr"
        output = runMCMC(
            model,
            phenotypes;
            chain_length=6,
            burnin=1,
            output_samples_frequency=5,
            outputEBV=false,
            output_heritability=false,
            output_marker_effect_samples=false,
            output_folder=output_folder,
            fast_blocks=[1, 3],
            memory_guard=:off,
            printout_model_info=false,
            seed=782,
        )
        result = output["annotation heritability enrichment annotation_enrichment_bayesr_geno"]
        @test nrow(result) == 2
        @test all(result.Posterior_Iterations .== 5)
        @test all(0 .< result.Posterior_Samples .<= 5)
        @test all(isfinite, result.Enrichment_Estimate)
        @test all(result.Allocation_Method .== "marginal_full_counting")
        rm(output_folder; recursive=true)
    end
end
