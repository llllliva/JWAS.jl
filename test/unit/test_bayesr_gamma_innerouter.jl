using Test
using JWAS
using CSV
using DataFrames
using JWAS.Datasets

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
            output = runMCMC(
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
                fast_blocks=[1, 3, 5],
                independent_blocks=false,
                inner_outer_diagnostics=true,
                BayesR_gamma=gamma,
            )

            marker_output = output["marker effects bayesr_gamma_innerouter_geno"]
            @test model.MCMCinfo.bayesr_gamma == gamma
            @test all(Symbol(name) in propertynames(marker_output) for name in [
                "InnerOuter_Samples",
                "InnerOuter_Model_Frequency",
                "InnerOuter_Class1_PIP",
                "InnerOuter_Class2_PIP",
                "InnerOuter_Class3_PIP",
                "InnerOuter_Class4_PIP",
            ])
            @test all(marker_output.InnerOuter_Samples .> 0)

            class_variances = output["BayesR class variances bayesr_gamma_innerouter_geno"]
            @test class_variances.Gamma == gamma
            @test class_variances.Estimate[1] == 0.0
            @test all(class_variances.Estimate[2:end] .>= 0.0)
            @test isfile("results/BayesR_class_variances_bayesr_gamma_innerouter_geno.txt")
            @test isfile("results/MCMC_samples_bayesr_class_variances_bayesr_gamma_innerouter_geno.txt")
            @test isfile("results/marker_effects_bayesr_gamma_innerouter_geno.txt")
        end
    end
end
