using Test
using JWAS
using CSV
using DataFrames
using JWAS.Datasets

@testset "Single-trait BayesC inner/outer diagnostics" begin
    genofile = Datasets.dataset("genotypes.txt", dataset_name="demo_7animals")
    phenotypes = CSV.read(
        Datasets.dataset("phenotypes.txt", dataset_name="demo_7animals"),
        DataFrame,
        delim=',',
        missingstring=["NA"],
    )
    annotations = Float64[
        1 0
        1 1
        0 1
        1 0
        0 1
    ]

    global bayesc_innerouter_annotated = get_genotypes(
        genofile,
        1.0;
        method="BayesC",
        Pi=0.8,
        estimatePi=true,
        separator=',',
        quality_control=false,
        annotations=annotations,
    )
    model = build_model("y1 = intercept + bayesc_innerouter_annotated", 1.0)

    mktempdir() do output_folder
        output = runMCMC(
            model,
            phenotypes;
            chain_length=6,
            burnin=2,
            output_samples_frequency=2,
            output_folder=output_folder,
            seed=123,
            outputEBV=false,
            output_heritability=false,
            output_annotation_enrichment=false,
            output_marker_effect_samples=false,
            printout_model_info=false,
            fast_blocks=[1, 3, 5],
            independent_blocks=false,
            inner_outer_diagnostics=true,
            memory_guard=:off,
        )

        marker_output = output["marker effects bayesc_innerouter_annotated"]
        @test all(Symbol(name) in propertynames(marker_output) for name in [
            "InnerOuter_Samples",
            "InnerOuter_Estimate",
            "InnerOuter_SD",
            "InnerOuter_Model_Frequency",
        ])
        @test marker_output.InnerOuter_Samples == UInt64[8, 8, 8, 8, 4]
        @test all(isfinite, marker_output.InnerOuter_Model_Frequency)
        @test all(0.0 .<= marker_output.InnerOuter_Model_Frequency .<= 1.0)
        @test all(isfinite, marker_output.InnerOuter_Estimate)
        @test all(isfinite, marker_output.InnerOuter_SD)

        summary = JWAS.summarize_bayesr_inner_outer(model.M[1].innerOuterStats)
        @test summary.inclusion_pip == marker_output.InnerOuter_Model_Frequency
        @test all(isapprox.(vec(sum(summary.class_pip, dims=2)), 1.0))
        @test isfile(joinpath(
            model.MCMCinfo.output_folder,
            "marker_effects_bayesc_innerouter_annotated.txt",
        ))
    end

    global bayesc_innerouter_invalid = get_genotypes(
        genofile,
        1.0;
        method="BayesC",
        Pi=0.8,
        estimatePi=true,
        separator=',',
        quality_control=false,
    )
    invalid_model = build_model("y1 = intercept + bayesc_innerouter_invalid", 1.0)
    mktempdir() do output_folder
        @test_throws ErrorException runMCMC(
            invalid_model,
            phenotypes;
            chain_length=4,
            burnin=1,
            output_samples_frequency=1,
            output_folder=output_folder,
            outputEBV=false,
            output_heritability=false,
            printout_model_info=false,
            fast_blocks=false,
            inner_outer_diagnostics=true,
            memory_guard=:off,
        )
    end
end
