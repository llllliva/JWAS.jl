using Test
using JWAS
using CSV
using DataFrames
using JWAS.Datasets
using LinearAlgebra
using Random

@testset "BayesR collapsed pi helpers" begin
    @test JWAS.validate_bayesr_pi_sampler(:standard) === :standard
    @test JWAS.validate_bayesr_pi_sampler(" COLLAPSED ") === :collapsed
    @test_throws ErrorException JWAS.validate_bayesr_pi_sampler(:other)
    @test_throws ErrorException JWAS.validate_bayesr_pi_sampler(true)

    delta = Int[1, 1, 2, 4, 1]
    @test JWAS.bayesr_class_counts(delta, 4) == [3, 1, 0, 1]
    @test_throws ErrorException JWAS.bayesr_class_counts(Int[1, 5], 4)

    counts_without_marker = [7, 2, 1, 0]
    pi = [0.94, 0.049, 0.01, 0.001]
    collapsed_weights = exp.([
        JWAS.bayesr_log_class_prior(:collapsed, pi, counts_without_marker, classi)
        for classi in 1:4
    ])
    @test collapsed_weights ≈ [8.0, 3.0, 2.0, 1.0]
    standard_weights = exp.([
        JWAS.bayesr_log_class_prior(:standard, pi, nothing, classi)
        for classi in 1:4
    ])
    @test standard_weights ≈ pi
end

@testset "BayesR standard sampler remains the seeded default" begin
    X = Float64[
        0 2
        1 1
        2 0
        1 1
    ]
    x_array = [collect(view(X, :, marker)) for marker in 1:size(X, 2)]
    xp_rinv_x = [dot(x, x) for x in x_array]
    pi = Float64[0.94, 0.049, 0.01, 0.001]
    gamma = Float64[0.0, 0.01, 0.1, 1.0]

    y_default = Float64[0.8, -0.1, 0.3, 0.5]
    alpha_default = zeros(Float64, 2)
    delta_default = ones(Int, 2)
    Random.seed!(20260803)
    JWAS.BayesR!(x_array, x_array, xp_rinv_x,
                 y_default, alpha_default, delta_default,
                 1.0, 0.2, pi, gamma)

    y_explicit = Float64[0.8, -0.1, 0.3, 0.5]
    alpha_explicit = zeros(Float64, 2)
    delta_explicit = ones(Int, 2)
    Random.seed!(20260803)
    JWAS.BayesR!(x_array, x_array, xp_rinv_x,
                 y_explicit, alpha_explicit, delta_explicit,
                 1.0, 0.2, pi, gamma;
                 pi_sampler=:standard)

    @test delta_default == delta_explicit
    @test alpha_default == alpha_explicit
    @test y_default == y_explicit
end

@testset "BayesR collapsed pi production path" begin
    genofile = Datasets.dataset("genotypes.txt", dataset_name="demo_7animals")
    phenotypes = CSV.read(
        Datasets.dataset("phenotypes.txt", dataset_name="demo_7animals"),
        DataFrame,
        delim=',',
        missingstring=["NA"],
    )

    mktempdir() do tmpdir
        cd(tmpdir) do
            global collapsed_dense_geno = get_genotypes(
                genofile,
                1.0;
                method="BayesR",
                Pi=[0.94, 0.049, 0.01, 0.001],
                estimatePi=true,
                separator=',',
                quality_control=false,
            )
            dense_model = build_model("y1 = intercept + collapsed_dense_geno", 1.0)
            dense_output = runMCMC(
                dense_model,
                phenotypes;
                chain_length=20,
                burnin=5,
                output_samples_frequency=5,
                output_folder="collapsed_dense",
                output_marker_effect_samples=false,
                seed=123,
                outputEBV=false,
                output_heritability=false,
                printout_model_info=false,
                BayesR_pi_sampler=:collapsed,
            )
            @test dense_model.MCMCinfo.bayesr_pi_sampler === :collapsed
            @test haskey(dense_output, "pi_collapsed_dense_geno")
            @test isapprox(sum(dense_output["pi_collapsed_dense_geno"].Estimate), 1.0; atol=1e-6)
            @test all(0.0 .<= dense_output["marker effects collapsed_dense_geno"].Model_Frequency .<= 1.0)

            global collapsed_block_geno = get_genotypes(
                genofile,
                1.0;
                method="BayesR",
                Pi=[0.94, 0.049, 0.01, 0.001],
                estimatePi=true,
                separator=',',
                quality_control=false,
            )
            block_model = build_model("y1 = intercept + collapsed_block_geno", 1.0)
            block_output = runMCMC(
                block_model,
                phenotypes;
                chain_length=12,
                burnin=2,
                output_samples_frequency=2,
                output_folder="collapsed_block",
                output_marker_effect_samples=false,
                seed=456,
                outputEBV=false,
                output_heritability=false,
                printout_model_info=false,
                fast_blocks=[1, 3, 5],
                BayesR_inner_iterations=1,
                BayesR_pi_sampler="collapsed",
            )
            @test block_model.MCMCinfo.bayesr_pi_sampler === :collapsed
            @test haskey(block_output, "pi_collapsed_block_geno")
        end
    end
end

@testset "BayesR collapsed pi validation" begin
    genofile = Datasets.dataset("genotypes.txt", dataset_name="demo_7animals")
    phenotypes = CSV.read(
        Datasets.dataset("phenotypes.txt", dataset_name="demo_7animals"),
        DataFrame,
        delim=',',
        missingstring=["NA"],
    )

    global collapsed_fixed_pi_geno = get_genotypes(
        genofile,
        1.0;
        method="BayesR",
        Pi=[0.94, 0.049, 0.01, 0.001],
        estimatePi=false,
        separator=',',
        quality_control=false,
    )
    fixed_model = build_model("y1 = intercept + collapsed_fixed_pi_geno", 1.0)
    fixed_outdir = tempname()
    fixed_error = try
        runMCMC(
            fixed_model,
            phenotypes;
            chain_length=10,
            output_folder=fixed_outdir,
            outputEBV=false,
            output_heritability=false,
            printout_model_info=false,
            BayesR_pi_sampler=:collapsed,
        )
        nothing
    catch error
        error
    end
    @test fixed_error isa ErrorException
    @test occursin("estimatePi=true", sprint(showerror, fixed_error))
    @test !isdir(fixed_outdir)
end
