using Test
using JWAS
using DataFrames
using CSV
using Statistics

@testset "Shared dense BayesC classes" begin
    source_matrix = Float32[
        0 1 2 0
        1 1 1 2
        2 0 0 1
        1 2 1 0
    ]
    source = get_genotypes(
        source_matrix,
        1.0;
        method="BayesC",
        quality_control=false,
        center=false,
        copy_genotypes=false,
        Pi=0.9,
    )
    memberships = Bool[
        1 0
        1 1
        0 1
        0 0
    ]

    classes = get_shared_genotype_classes(
        source,
        memberships;
        add_unassigned=true,
        Pi=0.9,
    )

    @test length(classes) == 3
    @test [class.nMarkers for class in classes] == [2, 2, 1]
    @test classes[1].markerID == ["1", "2"]
    @test classes[2].markerID == ["2", "3"]
    @test classes[3].markerID == ["4"]
    @test all(class -> Base.mightalias(class.genotypes, source.genotypes), classes)
    @test all(class -> parent(class.genotypes) === source.genotypes, classes)
    @test classes[1].genotypes[:, 2] == source.genotypes[:, 2]
    @test classes[2].genotypes[:, 1] == source.genotypes[:, 2]

    source.genotypes[1, 2] = 7.0f0
    @test classes[1].genotypes[1, 2] == 7.0f0
    @test classes[2].genotypes[1, 1] == 7.0f0

    classes[1].π = 0.8
    @test classes[2].π == 0.9

    @test_throws ErrorException get_shared_genotype_classes(
        source,
        Float64[1 0; 0.5 1; 0 1; 0 0],
    )
end


@testset "Shared dense classes support unphenotyped genotype rows" begin
    source_matrix = Float32[
        0 1 2 0
        1 1 1 2
        2 0 0 1
        1 2 1 0
        0 0 2 1
        2 1 0 2
    ]
    source = get_genotypes(
        source_matrix,
        1.0;
        method="BayesC",
        quality_control=false,
        center=true,
        copy_genotypes=false,
        Pi=0.8,
    )
    memberships = Bool[
        1 0
        1 1
        0 1
        0 0
    ]
    classes = get_shared_genotype_classes(
        source,
        memberships;
        add_unassigned=true,
        Pi=0.8,
    )

    global shared_subset_a = classes[1]
    global shared_subset_b = classes[2]
    global shared_subset_others = classes[3]
    phenotypes = DataFrame(
        ID=["1", "3", "4", "6"],
        y=[0.2, -0.4, 0.6, 1.0],
    )
    model = build_model(
        "y = intercept + shared_subset_a + shared_subset_b + shared_subset_others",
        1.0,
    )

    mktempdir() do output_folder
        output = runMCMC(
            model,
            phenotypes;
            chain_length=8,
            burnin=2,
            output_samples_frequency=2,
            output_folder=output_folder,
            outputEBV=false,
            output_heritability=true,
            output_marker_effect_samples=false,
            printout_frequency=99,
            printout_model_info=false,
            fast_blocks=false,
            memory_guard=:off,
            seed=123,
        )

        @test haskey(output, "marker effects shared_subset_a")
        @test haskey(output, "marker effects shared_subset_b")
        @test haskey(output, "marker effects shared_subset_others")
        @test isfile(joinpath(
            model.MCMCinfo.output_folder,
            "MCMC_samples_class_heritability.txt",
        ))
    end
end

@testset "Shared dense classes run through ordinary multiclass MCMC" begin
    source_matrix = Float32[
        0 1 2
        1 1 1
        2 0 0
        1 2 1
    ]
    source = get_genotypes(
        source_matrix,
        1.0;
        method="BayesC",
        quality_control=false,
        center=true,
        copy_genotypes=false,
        Pi=0.8,
    )
    memberships = Bool[
        1 0
        1 1
        0 1
    ]
    classes = get_shared_genotype_classes(
        source,
        memberships;
        add_unassigned=false,
        Pi=0.8,
    )

    global shared_dense_class_a = classes[1]
    global shared_dense_class_b = classes[2]
    phenotypes = DataFrame(ID=["3", "1", "4", "2"], y=[-0.4, 0.2, 0.6, 1.0])

    mktempdir() do output_folder
        model = build_model("y = intercept + shared_dense_class_a + shared_dense_class_b", 1.0)
        model.M[1].α = [Float32[0.4, -0.2]]
        model.M[2].α = [Float32[0.1, 0.3]]
        deterministic_h2 = JWAS.multiclass_bayesc_class_heritability(model, 2.0)
        deterministic_total = model.M[1].genotypes * model.M[1].α[1] +
                              model.M[2].genotypes * model.M[2].α[1]
        expected_total_h2 = var(deterministic_total) / (var(deterministic_total) + 2.0)
        @test isapprox(sum(deterministic_h2), expected_total_h2; rtol=1e-6)
        model.M[1].α = false
        model.M[2].α = false

        output = runMCMC(
            model,
            phenotypes;
            chain_length=6,
            burnin=2,
            output_samples_frequency=2,
            output_folder=output_folder,
            outputEBV=false,
            output_heritability=true,
            output_marker_effect_samples=false,
            printout_frequency=99,
            printout_model_info=false,
            fast_blocks=[1, 2],
            memory_guard=:off,
            seed=941,
        )

        @test haskey(output, "marker effects shared_dense_class_a")
        @test haskey(output, "marker effects shared_dense_class_b")
        @test Base.mightalias(model.M[1].genotypes, source.genotypes)
        @test Base.mightalias(model.M[2].genotypes, source.genotypes)

        actual_output_folder = model.MCMCinfo.output_folder
        class_h2_file = joinpath(actual_output_folder, "MCMC_samples_class_heritability.txt")
        @test isfile(class_h2_file)
        class_h2_samples = CSV.read(class_h2_file, DataFrame)
        @test names(class_h2_samples) == ["shared_dense_class_a", "shared_dense_class_b"]
        @test nrow(class_h2_samples) == 2
        class_h2_matrix = Matrix(class_h2_samples)
        valid_rows = vec(all(isfinite, class_h2_matrix; dims=2))
        @test any(valid_rows)
        total_h2_samples = vec(sum(class_h2_matrix[valid_rows, :], dims=2))
        @test all((total_h2_samples .> 0.0) .& (total_h2_samples .< 1.0))
        @test !any(startswith(file, "MCMC_samples_EBV_") for file in readdir(actual_output_folder))
    end
end
