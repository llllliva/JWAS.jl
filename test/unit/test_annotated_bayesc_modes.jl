using Test
using JWAS
using DataFrames
using CSV

function write_annotated_test_geno(path::AbstractString)
    geno = DataFrame(
        ID = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        m1 = [0, 1, 2, 0, 1, 2, 1],
        m2 = [1, 0, 1, 2, 1, 0, 2],
        m3 = [2, 1, 0, 1, 2, 1, 0],
        m4 = [0, 2, 1, 1, 0, 2, 1],
        m5 = [1, 1, 2, 0, 2, 0, 1],
    )
    CSV.write(path, geno)
end

function annotated_test_phenotypes()
    return DataFrame(
        ID = ["a1", "a2", "a3", "a4", "a5", "a6", "a7"],
        y1 = Float32[1.2, -0.4, 0.8, -1.0, 0.6, -0.2, 0.3],
    )
end

function annotated_test_matrix()
    return [
        0.0 1.0
        1.0 0.0
        1.0 1.0
        0.0 0.5
        0.5 0.0
    ]
end

function run_single_trait_mcmc(geno, phenotypes, outdir; fast_blocks=false)
    model = build_model("y1 = intercept + geno", 1.0)
    return runMCMC(model, phenotypes;
                   chain_length=40,
                   burnin=10,
                   output_samples_frequency=5,
                   output_folder=outdir,
                   outputEBV=false,
                   output_heritability=false,
                   seed=20260314,
                   fast_blocks=fast_blocks,
                   memory_guard=:off)
end

@testset "Annotated BayesC modes" begin
    mktempdir() do tmpdir
        cd(tmpdir) do
            write_annotated_test_geno("geno.csv")
            phenotypes = annotated_test_phenotypes()
            annotations = annotated_test_matrix()

            @testset "annotations only (dense)" begin
                global geno = get_genotypes("geno.csv", 1.0;
                                            separator=',',
                                            header=true,
                                            method="BayesC",
                                            quality_control=false,
                                            annotations=annotations)
                out = run_single_trait_mcmc(geno, phenotypes, "annot_dense")
                @test haskey(out, "annotation coefficients geno")
                @test size(out["annotation coefficients geno"], 1) == 3
                @test out["annotation coefficients geno"].Annotation[1] == "Intercept"
            end

            @testset "fast_blocks only (dense)" begin
                global geno = get_genotypes("geno.csv", 1.0;
                                            separator=',',
                                            header=true,
                                            method="BayesC",
                                            quality_control=false)
                out = run_single_trait_mcmc(geno, phenotypes, "fast_blocks_dense"; fast_blocks=true)
                @test haskey(out, "marker effects geno")
            end

            @testset "annotations plus fast_blocks (dense)" begin
                global geno = get_genotypes("geno.csv", 1.0;
                                            separator=',',
                                            header=true,
                                            method="BayesC",
                                            quality_control=false,
                                            annotations=annotations)
                out = run_single_trait_mcmc(geno, phenotypes, "annot_fast_blocks_dense"; fast_blocks=true)
                @test haskey(out, "annotation coefficients geno")
            end

            @testset "storage=:stream only" begin
                prefix = JWAS.prepare_streaming_genotypes("geno.csv";
                                                          separator=',',
                                                          header=true,
                                                          quality_control=false,
                                                          center=true)
                global geno = get_genotypes(prefix, 1.0;
                                            method="BayesC",
                                            storage=:stream)
                out = run_single_trait_mcmc(geno, phenotypes, "stream_only")
                @test haskey(out, "marker effects geno")
                @test !haskey(out, "annotation coefficients geno")
            end

            @testset "annotations plus storage=:stream" begin
                prefix = JWAS.prepare_streaming_genotypes("geno.csv";
                                                          separator=',',
                                                          header=true,
                                                          quality_control=false,
                                                          center=true)
                global geno = get_genotypes(prefix, 1.0;
                                            method="BayesC",
                                            storage=:stream,
                                            annotations=annotations)
                out = run_single_trait_mcmc(geno, phenotypes, "annot_stream")
                @test haskey(out, "annotation coefficients geno")
                @test size(out["annotation coefficients geno"], 1) == 3
            end

            @testset "annotations plus fast_blocks plus storage=:stream is rejected" begin
                prefix = JWAS.prepare_streaming_genotypes("geno.csv";
                                                          separator=',',
                                                          header=true,
                                                          quality_control=false,
                                                          center=true)
                global geno = get_genotypes(prefix, 1.0;
                                            method="BayesC",
                                            storage=:stream,
                                            annotations=annotations)
                model = build_model("y1 = intercept + geno", 1.0)
                @test_throws ErrorException runMCMC(model, phenotypes;
                                                    chain_length=40,
                                                    burnin=10,
                                                    output_samples_frequency=5,
                                                    output_folder="annot_stream_fast_blocks",
                                                    outputEBV=false,
                                                    output_heritability=false,
                                                    seed=20260314,
                                                    fast_blocks=true,
                                                    memory_guard=:off)
            end
        end
    end
end
