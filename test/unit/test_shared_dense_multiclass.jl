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

@testset "Unequal shared classes with fast blocks" begin
    # Overlap, unassigned singleton, noncontiguous columns and subset rows.
    for block_spec in (true, 2, 20, [1]), weighted in (false, true)
        source = get_genotypes(Float32[mod(i*j+i+j+div(i,3), 3) for i in 1:12, j in 1:8],
            1.0; method="BayesC", quality_control=false, center=true,
            copy_genotypes=false, Pi=0.5)
        memberships = falses(8, 2)
        memberships[[1,3,5], 1] .= true
        memberships[[2,3,4,6,7], 2] .= true
        classes = get_shared_genotype_classes(source, memberships; add_unassigned=true, Pi=0.5)
        global fast_class_a, fast_class_b, fast_class_u = classes
        model = build_model("y = intercept + fast_class_a + fast_class_b + fast_class_u", 1.0)
        df = DataFrame(ID=string.([1,3,4,6,7,9,10,12]),
                       y=[0.2,-0.4,0.6,1.0,-0.8,0.3,0.7,-0.2])
        if weighted
            df.weights = [1.0,2.0,1.5,0.8,1.0,1.2,0.9,1.1]
        end
        mktempdir() do folder
            out = runMCMC(model, df; chain_length=160, burnin=2,
                output_samples_frequency=2, output_folder=folder,
                outputEBV=false, output_heritability=true,
                output_marker_effect_samples=false, printout_frequency=999,
                printout_model_info=false, fast_blocks=block_spec,
                inner_outer_diagnostics=true, memory_guard=:off, seed=142,
                heterogeneous_residuals=weighted)
            for Mi in model.M
                blocks = Mi.MArray
                @test sum(size(b, 2) for b in blocks) == Mi.nMarkers
                @test hcat(blocks...) == Mi.genotypes
                @test all(b -> Base.mightalias(b, source.genotypes), blocks)
                @test all(isfinite, Mi.α[1])
                @test haskey(out, "marker effects $(Mi.name)")
                marker_out = out["marker effects $(Mi.name)"]
                widths = vcat([fill(size(b,2), size(b,2)) for b in blocks]...)
                @test marker_out.InnerOuter_Samples == (model.MCMCinfo.chain_length - 2) .* widths
                @test all(0 .<= marker_out.InnerOuter_Model_Frequency .<= 1)
            end
            @test isfile(joinpath(model.MCMCinfo.output_folder, "MCMC_samples_class_heritability.txt"))
        end
    end
end

@testset "Class-local block validation" begin
    terms = [(name="a", nMarkers=3), (name="b", nMarkers=5), (name="u", nMarkers=1)]
    starts = JWAS.resolve_multiclass_block_starts(terms, 2)
    @test starts == Dict("a"=>[1,3], "b"=>[1,3,5], "u"=>[1])
    @test JWAS.fast_block_starts_for(starts, terms[2]) == [1,3,5]
    @test_throws ErrorException JWAS.resolve_multiclass_block_starts(terms, [1,2])
end

@testset "Sequential multiclass block residual consistency" begin
    using LinearAlgebra
    using Random
    Random.seed!(79)
    parent_X = randn(Float32, 10, 8)
    rows = [1,3,4,7,8,10]
    matrices = [view(parent_X, rows, cols) for cols in ([1,3,5], [2,3,4,6,7], [8])]
    for weights in (ones(Float32, 6), Float32[1,2,0.5,1,0.8,1.2])
        effects = [randn(Float32, size(X,2)) for X in matrices]
        y = randn(Float32, 6)
        residual = y - sum(X*a for (X,a) in zip(matrices, effects))
        for c in eachindex(matrices)
            X, a = matrices[c], effects[c]
            mats = JWAS.GibbsMats(X, weights; fast_blocks=collect(1:2:size(X,2)))
            JWAS.BayesABC_block!(mats.XArray, mats.xpRinvx, X, mats.XpRinvX,
                residual, a, copy(a), ones(Float32, length(a)),
                1.0f0, ones(Float32, length(a)), 0.5, weights)
            @test residual ≈ y - sum(Xc*ac for (Xc,ac) in zip(matrices, effects)) atol=2f-5
        end
    end
end
