# Unit tests for outputEBV(), getEBV(), and output functions
using Test, JWAS, DataFrames, CSV, JWAS.Datasets

phenofile = Datasets.dataset("phenotypes.txt", dataset_name="demo_7animals")
genofile = Datasets.dataset("genotypes.txt", dataset_name="demo_7animals")
phenotypes = CSV.read(phenofile, DataFrame, delim=',', missingstring=["NA"])

function write_stream_output_genotypes(path::AbstractString)
    open(path, "w") do io
        println(io, "ID,m1,m2,m3,m4")
        println(io, "a1,0,1,2,0")
        println(io, "a2,1,0,1,2")
        println(io, "a3,2,1,0,1")
        println(io, "a4,0,2,1,0")
        println(io, "a5,1,1,2,2")
        println(io, "a6,2,0,0,1")
    end
end

@testset "outputEBV and EBV results" begin
    @testset "EBV output with genotypes" begin
        global geno = get_genotypes(genofile, 1.0, separator=',', method="BayesC")
        model = build_model("y1 = intercept + geno", 1.0)
        outputEBV(model, geno.obsID)

        output = runMCMC(model, phenotypes,
                        chain_length=100,
                        burnin=20,
                        output_samples_frequency=10,
                        outputEBV=true,
                        output_folder="test_ebv_output",
                        seed=123)

        @test haskey(output, "EBV_y1")
        ebv_df = output["EBV_y1"]
        @test :ID in propertynames(ebv_df)
        @test :EBV in propertynames(ebv_df)
        @test :PEV in propertynames(ebv_df)
        @test size(ebv_df, 1) > 0
        @test all(ebv_df.PEV .>= 0)  # PEV should be non-negative

        rm("test_ebv_output", recursive=true)
    end

    @testset "EBV output with heritability" begin
        global geno = get_genotypes(genofile, 1.0, separator=',', method="RR-BLUP")
        model = build_model("y1 = intercept + geno", 1.0)

        output = runMCMC(model, phenotypes,
                        chain_length=100,
                        burnin=20,
                        output_samples_frequency=10,
                        outputEBV=true,
                        output_heritability=true,
                        output_folder="test_ebv_h2",
                        seed=123)

        @test haskey(output, "EBV_y1")
        @test haskey(output, "heritability")
        @test haskey(output, "genetic_variance")

        h2 = output["heritability"]
        @test :Estimate in propertynames(h2)
        @test all(0 .<= h2.Estimate .<= 1)

        rm("test_ebv_h2", recursive=true)
    end
end

@testset "outputMCMCsamples for location parameters" begin
    global geno = get_genotypes(genofile, 1.0, separator=',', method="BayesC")
    model = build_model("y1 = intercept + geno", 1.0)
    outputMCMCsamples(model, "intercept")

    output = runMCMC(model, phenotypes,
                    chain_length=50,
                    output_samples_frequency=10,
                    output_folder="test_mcmc_samples",
                    seed=123)

    @test isfile("test_mcmc_samples/MCMC_samples_y1.intercept.txt")
    rm("test_mcmc_samples", recursive=true)
end

@testset "stream EBV output uses train/output views" begin
    mktempdir() do tmpdir
        cd(tmpdir) do
            write_stream_output_genotypes("geno.csv")
            prefix = JWAS.prepare_streaming_genotypes("geno.csv";
                                                      separator=',',
                                                      header=true,
                                                      quality_control=false,
                                                      center=true)

            pheno_subset = DataFrame(ID=["a3", "a1", "a5", "a2"],
                                     y1=Float32[0.8, 1.1, 0.5, -0.3])
            output_ids = ["a6", "a3", "a1", "a5"]

            global geno = get_genotypes("geno.csv", 1.0;
                                        separator=',',
                                        header=true,
                                        method="BayesC",
                                        quality_control=false,
                                        center=true)
            model_dense = build_model("y1 = intercept + geno", 1.0)
            outputEBV(model_dense, output_ids)
            dense = runMCMC(model_dense, pheno_subset;
                            chain_length=40,
                            burnin=10,
                            output_samples_frequency=10,
                            outputEBV=true,
                            output_heritability=false,
                            output_folder="dense_stream_ebv_dense",
                            output_marker_parameter_samples=false,
                            seed=2026,
                            memory_guard=:off)

            global geno = get_genotypes(prefix, 1.0;
                                        method="BayesC",
                                        storage=:stream)
            model_stream = build_model("y1 = intercept + geno", 1.0)
            outputEBV(model_stream, output_ids)
            stream = runMCMC(model_stream, pheno_subset;
                             chain_length=40,
                             burnin=10,
                             output_samples_frequency=10,
                             outputEBV=true,
                             output_heritability=false,
                             output_folder="dense_stream_ebv_stream",
                             output_marker_parameter_samples=false,
                             seed=2026,
                             memory_guard=:off)

            @test collect(dense["EBV_y1"].ID) == output_ids
            @test collect(stream["EBV_y1"].ID) == output_ids
            @test Vector{Float64}(stream["EBV_y1"].EBV) ≈ Vector{Float64}(dense["EBV_y1"].EBV) atol=1e-4
            @test Vector{Float64}(stream["EBV_y1"].PEV) ≈ Vector{Float64}(dense["EBV_y1"].PEV) atol=1e-4

            @test isfile("dense_stream_ebv_stream/MCMC_samples_EBV_y1.bin")
            @test isfile("dense_stream_ebv_stream/MCMC_samples_EBV_y1.ids.txt")
            @test isfile("dense_stream_ebv_stream/MCMC_samples_EBV_y1.meta")
            @test !isfile("dense_stream_ebv_stream/MCMC_samples_EBV_y1.txt")

            ids_bin, samples_bin = readEBVsamples("dense_stream_ebv_stream", "y1"; mmap=false)
            @test ids_bin == output_ids
            @test size(samples_bin) == (3, length(output_ids))

            ids_mmap, samples_mmap = readEBVsamples("dense_stream_ebv_stream", "y1"; mmap=true)
            @test ids_mmap == output_ids
            @test size(samples_mmap) == (3, length(output_ids))
            @test Array(samples_mmap) ≈ samples_bin atol=1e-6

            @test !isfile("dense_stream_ebv_stream/MCMC_samples_marker_effects_geno_y1.txt")
            @test !isfile("dense_stream_ebv_stream/MCMC_samples_marker_effects_variances_geno.txt")
            @test !isfile("dense_stream_ebv_stream/MCMC_samples_pi_geno.txt")
        end
    end
end
