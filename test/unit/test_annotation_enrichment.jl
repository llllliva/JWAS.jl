using Test
using JWAS
using DataFrames

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
        genotype.α = [Float32[1.0, 2.0, 0.5, 0.0]]
        genotype.annotations.enrichment_stats = JWAS.initialize_annotation_enrichment(genotype)

        JWAS.update_annotation_enrichment!(genotype, 2.0)
        result = JWAS.annotation_enrichment_dataframe(genotype)

        twopq = vec(2.0 .* genotype.alleleFreq .* (1.0 .- genotype.alleleFreq))
        marker_variance = twopq .* Float64.(genotype.α[1]).^2
        group_variance = vec(ANNOTATION_ENRICHMENT_A' * marker_variance)
        total_variance = sum(marker_variance)
        expected_proportion = group_variance ./ total_variance
        expected_h2 = group_variance ./ (total_variance + 2.0)
        expected_enrichment = expected_proportion ./ vec(sum(ANNOTATION_ENRICHMENT_A, dims=1) ./ 4)

        @test result.Heritability_Estimate ≈ expected_h2
        @test result.H2_Proportion_Estimate ≈ expected_proportion
        @test result.Enrichment_Estimate ≈ expected_enrichment
        @test result.Posterior_Iterations == [1, 1]
        @test result.Posterior_Samples == [1, 1]
        @test result.SNP_Count == [2, 2]
        @test all(result.Heritability_SD .== 0.0)
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
                output_annotation_enrichment=true,
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
            @test nrow(result) == 2
            @test all(result.Posterior_Iterations .== 6)
            @test all(0 .< result.Posterior_Samples .<= 6)
            @test all(isfinite, result.Heritability_Estimate)
            @test all(isfinite, result.H2_Proportion_Estimate)
            @test all(isfinite, result.Enrichment_Estimate)
            @test all(result.Heritability_Estimate .>= 0.0)
            @test isfile(joinpath(output_folder, replace(key, " " => "_") * ".txt"))
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
            output_annotation_enrichment=true,
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
        rm(output_folder; recursive=true)
    end
end
