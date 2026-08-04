using Test
using JWAS
using CSV
using DataFrames
using JWAS.Datasets

@testset "BayesR runtime option resolution" begin
    empty_env = Dict{String,String}()
    @test JWAS.resolve_bayesr_fast_blocks(false; env=empty_env) === false
    @test JWAS.resolve_bayesr_fast_blocks(true; env=empty_env) === true
    @test JWAS.resolve_bayesr_fast_blocks(true; env=Dict("BAYESR_BLOCK_STARTS" => "1, 3,5")) == [1, 3, 5]
    @test JWAS.resolve_bayesr_fast_blocks([1, 4]; env=Dict("BAYESR_BLOCK_STARTS" => "1,3,5")) == [1, 4]
    @test_throws ErrorException JWAS.resolve_bayesr_fast_blocks(true; env=Dict("BAYESR_BLOCK_STARTS" => "2,4"))
    @test_throws ErrorException JWAS.resolve_bayesr_fast_blocks(true; env=Dict("BAYESR_BLOCK_STARTS" => "1,x"))

    @test JWAS.resolve_bayesr_inner_iterations(false; env=empty_env) === false
    @test JWAS.resolve_bayesr_inner_iterations(false; env=Dict("BAYESR_INNER_ITERATIONS" => "auto")) === false
    @test JWAS.resolve_bayesr_inner_iterations(false; env=Dict("BAYESR_INNER_ITERATIONS" => "3")) == 3
    @test JWAS.resolve_bayesr_inner_iterations(4; env=Dict("BAYESR_INNER_ITERATIONS" => "3")) == 4
    @test_throws ErrorException JWAS.resolve_bayesr_inner_iterations(true; env=empty_env)
    @test_throws ErrorException JWAS.resolve_bayesr_inner_iterations(false; env=Dict("BAYESR_INNER_ITERATIONS" => "0"))
end

@testset "BayesR gamma input and inner/outer diagnostics" begin
    genofile = Datasets.dataset("genotypes.txt", dataset_name="demo_7animals")
    phenotypes = CSV.read(
        Datasets.dataset("phenotypes.txt", dataset_name="demo_7animals"),
        DataFrame,
        delim=',',
        missingstring=["NA"],
    )
    gamma = [0.0, 0.001, 0.1, 1.0]

    global bayesr_gamma_innerouter_geno = get_genotypes(
        genofile,
        1.0;
        method="BayesR",
        Pi=[0.94, 0.049, 0.01, 0.001],
        separator=',',
        quality_control=false,
    )
    model = build_model("y1 = intercept + bayesr_gamma_innerouter_geno", 1.0)

    mktempdir() do tmpdir
        cd(tmpdir) do
            output = withenv(
                "BAYESR_BLOCK_STARTS" => "1,3,5",
                "BAYESR_INNER_ITERATIONS" => "3",
            ) do
                runMCMC(
                    model,
                    phenotypes;
                    chain_length=12,
                    burnin=2,
                    output_samples_frequency=1,
                    output_folder="results",
                    seed=123,
                    outputEBV=false,
                    output_heritability=false,
                    printout_model_info=false,
                    fast_blocks=true,
                    independent_blocks=false,
                    inner_outer_diagnostics=true,
                    BayesR_gamma=gamma,
                    BayesR_inner_iterations=false,
                )
            end

            marker_output = output["marker effects bayesr_gamma_innerouter_geno"]
            @test model.MCMCinfo.bayesr_gamma == gamma
            @test model.MCMCinfo.fast_blocks == [1, 3, 5]
            @test model.MCMCinfo.bayesr_inner_iterations == 3
            @test all(Symbol(name) in propertynames(marker_output) for name in [
                "InnerOuter_Samples",
                "InnerOuter_Model_Frequency",
                "InnerOuter_Class1_PIP",
                "InnerOuter_Class2_PIP",
                "InnerOuter_Class3_PIP",
                "InnerOuter_Class4_PIP",
            ])
            @test all(marker_output.InnerOuter_Samples .> 0)
            @test all(marker_output.InnerOuter_Samples .== 30)  # 10 retained outer × 3 inner

            class_variances = output["BayesR class variances bayesr_gamma_innerouter_geno"]
            @test class_variances.Gamma == gamma
            @test class_variances.Estimate[1] == 0.0
            @test all(class_variances.Estimate[2:end] .>= 0.0)
            @test isfile("results/BayesR_class_variances_bayesr_gamma_innerouter_geno.txt")
            @test isfile("results/MCMC_samples_bayesr_class_variances_bayesr_gamma_innerouter_geno.txt")
            @test isfile("results/MCMC_samples_delta_bayesr_gamma_innerouter_geno_y1.txt")
            delta_trace = readlines("results/MCMC_samples_delta_bayesr_gamma_innerouter_geno_y1.txt")
            @test length(delta_trace) == 11  # header plus 10 retained outer samples
            @test isfile("results/marker_effects_bayesr_gamma_innerouter_geno.txt")

            global bayesr_gamma_innerouter_auto_geno = get_genotypes(
                genofile,
                1.0;
                method="BayesR",
                Pi=[0.94, 0.049, 0.01, 0.001],
                separator=',',
                quality_control=false,
            )
            automatic_model = build_model(
                "y1 = intercept + bayesr_gamma_innerouter_auto_geno",
                1.0,
            )
            automatic_output = withenv(
                "BAYESR_BLOCK_STARTS" => "1,3,5",
                "BAYESR_INNER_ITERATIONS" => nothing,
            ) do
                runMCMC(
                    automatic_model,
                    phenotypes;
                    chain_length=12,
                    burnin=2,
                    output_samples_frequency=1,
                    output_folder="results_auto",
                    seed=123,
                    outputEBV=false,
                    output_heritability=false,
                    printout_model_info=false,
                    fast_blocks=true,
                    independent_blocks=false,
                    inner_outer_diagnostics=true,
                    BayesR_gamma=gamma,
                )
            end
            automatic_markers = automatic_output["marker effects bayesr_gamma_innerouter_auto_geno"]
            @test automatic_model.MCMCinfo.bayesr_inner_iterations === false
            @test automatic_markers.InnerOuter_Samples == UInt64[20, 20, 20, 20, 10]
        end
    end
end
