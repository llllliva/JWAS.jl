using Test, JWAS, DataFrames, CSV, Random, Statistics

@testset "Contiguous shared genotype selection" begin
    ids = ["animal_$i" for i in 1:12]
    @test JWAS.contiguous_genotype_rows(ids[1:8],ids) === 1:8
    @test JWAS.contiguous_genotype_rows(ids[5:12],ids) === 5:12
    @test_throws ErrorException JWAS.contiguous_genotype_rows(ids[[1,3]],ids)
    @test_throws ErrorException JWAS.contiguous_genotype_rows(ids[[2,1]],ids)
    @test_throws ErrorException JWAS.contiguous_genotype_rows(ids[[1,1]],ids)
    @test_throws ErrorException JWAS.contiguous_genotype_rows(["absent"],ids)
    @test_throws ErrorException JWAS.contiguous_genotype_rows(String[],ids)
    @test_throws ErrorException JWAS.contiguous_genotype_rows(ids,vcat(ids[1:11],ids[1]))
    for precision in (Float32,Float64)
        X = reshape(precision.(1:84),12,7)
        training = view(X,1:8,:)
        gibbs = JWAS.GibbsMats(training,ones(precision,8);fast_blocks=[1,4,7])
        @test all(block -> Base.mightalias(block,X),gibbs.XArray)
        @test size.(gibbs.XArray) == [(8,3),(8,3),(8,1)]
        @test gibbs.xRinvArray === gibbs.xArray
        @test all(block -> block isa StridedMatrix{precision},gibbs.XArray)
    end
end

function shared_genotype_case(shared,method,fast,seed; annotated=false,precision=Float32,
                         n=36,t=24,p=23,outer=300,burnin=100,freq=10,folder=nothing,
                         selected_train=collect(1:t),prediction_rows=collect(1:n),
                         weights=false,shared_override=shared)
    rng = MersenneTwister(20261021)
    X = precision.(rand(rng,0:2,n,p))
    ids = ["animal_$(lpad(i,4,'0'))" for i in 1:n]
    markers = ["rs_$(1000+7*i)" for i in 1:p]
    # Generate the same raw response before in-place adopted-matrix centering.
    observations = 0.3 .* Float64.(X[:,1]) .+ 0.1 .* collect(1:n)./n .+ randn(rng,n)
    pi = 0.65
    annotation = annotated ? Float64.(hcat(isodd.(1:p),mod.(1:p,3).==0)) : false
    input_options = shared_override === nothing ? (;) : (;copy_genotypes=!shared)
    global shared_genotype_test_genotypes = get_genotypes(X,1.0;method,Pi=pi,
        quality_control=false,double_precision=(precision===Float64),
        input_options...,annotations=annotation)
    g = shared_genotype_test_genotypes
    g.obsID .= ids; g.markerID .= markers
    owned = g.genotypes
    shared && @test g.genotypes === X
    train_set = Set(selected_train)
    ph = DataFrame(ID=ids,
        y=Union{Missing,Float64}[i in train_set ? observations[i] : missing for i in 1:n],
        FARM=ifelse.(iseven.(1:n),"A","B"),LITTER=["L$(mod1(i,6))" for i in 1:n])
    weights !== false && (ph.weights = weights)
    model = build_model("y = intercept + FARM + LITTER + shared_genotype_test_genotypes",1.0)
    set_random(model,"LITTER",1.0)
    outputEBV(model,ids[prediction_rows])
    outdir = folder === nothing ? joinpath(mktempdir(),"mcmc") : folder
    chain = fast === false ? outer : outer*fast
    alignment_options = shared_override === nothing ? (;) : (;shared_genotypes=shared_override)
    timing = @timed runMCMC(model,ph;chain_length=chain,burnin=(fast === false ? burnin : burnin*fast),output_samples_frequency=freq,
        fast_blocks=fast,alignment_options...,seed,output_folder=outdir,
        double_precision=(precision===Float64),heterogeneous_residuals=(weights!==false),
        output_heritability=false,printout_model_info=false,printout_frequency=outer+1)
    report = JWAS.genotype_storage_report(g)
    if shared
        @test report.backing_matrix_count == 1
        @test report.backing_matrix_bytes == sizeof(X)
        @test Base.mightalias(g.genotypes,owned)
        @test report.output_shares_training_parent
        @test report.all_blocks_share_training
        @test g.genotypes isa StridedMatrix{precision}
    else
        @test report.backing_matrix_count == (t < n ? 2 : 1)
    end
    @test all(ismissing,ph.y[setdiff(1:n,selected_train)])
    ebv = CSV.read(joinpath(outdir,"EBV_y.txt"),DataFrame;types=Dict(:ID=>String))
    marker = timing.value["marker effects shared_genotype_test_genotypes"]
    @test nrow(ebv) == length(prediction_rows)
    @test Set(String.(ebv.ID)) == Set(ids[prediction_rows])
    @test nrow(marker) == p
    @test Set(string.(marker.Marker_ID)) == Set(markers)
    @test length(readlines(joinpath(outdir,"MCMC_samples_residual_variance.txt")))-1 == fld(outer-burnin,freq)
    @test all(isfinite,ebv.EBV) && all(isfinite,ebv.PEV)
    return (;ebv,marker,report,timing,model,outdir)
end

function compare_shared_genotype_outputs(copied,shared)
    # Compare by explicit keys, never output position or lexicographic order.
    animal_keys = String.(copied.ebv.ID)
    @test length(unique(animal_keys)) == length(animal_keys)
    @test Set(animal_keys) == Set(String.(shared.ebv.ID))
    animals = innerjoin(copied.ebv,shared.ebv;on=:ID,makeunique=true,validate=(true,true))
    @test nrow(animals) == length(animal_keys)
    @test isapprox(animals.EBV,animals.EBV_1;atol=1e-4,rtol=1e-4)
    @test isapprox(animals.PEV,animals.PEV_1;atol=1e-4,rtol=1e-4)
    a,b = copy(copied.marker),copy(shared.marker)
    a.Marker_ID = string.(a.Marker_ID); b.Marker_ID = string.(b.Marker_ID)
    @test nrow(a) == nrow(b)
    @test length(unique(a.Marker_ID)) == nrow(a)
    @test Set(a.Marker_ID) == Set(b.Marker_ID)
    markers = innerjoin(a,b;on=[:Trait,:Marker_ID],makeunique=true,validate=(true,true))
    @test nrow(markers) == nrow(a)
    for name in (:Estimate,:SD,:Model_Frequency)
        @test isapprox(markers[!,name],markers[!,Symbol(string(name)*"_1")];atol=1e-4,rtol=1e-4)
    end
    (;animals,markers)
end

@testset "Production MCMC with one shared matrix" begin
    for seed in (73,123,456), precision in (Float32,Float64), method in ("BayesC",), fast in (false,5)
        copied = shared_genotype_case(false,method,fast,seed;precision)
        shared = shared_genotype_case(true,method,fast,seed;precision)
        compare_shared_genotype_outputs(copied,shared)
    end
    # Omitting the keyword must take the same sharing path as explicit true.
    default_shared = shared_genotype_case(true,"BayesC",5,123;shared_override=nothing)
    @test default_shared.report.training_is_view
    @test default_shared.report.backing_matrix_count == 1
    for method in ("BayesC",)
        copied = shared_genotype_case(false,method,5,123;annotated=true)
        shared = shared_genotype_case(true,method,5,123;annotated=true)
        compare_shared_genotype_outputs(copied,shared)
    end
    double = shared_genotype_case(true,"BayesC",5,123;precision=Float64)
    @test eltype(double.model.M[1].genotypes) === Float64
    subset = shared_genotype_case(true,"BayesC",5,123;prediction_rows=collect(25:36))
    @test nrow(subset.ebv) == 12
end

@testset "Shared matrix guards" begin
    @test_throws ErrorException shared_genotype_case(true,"BayesC",5,123;selected_train=[1,3,5])
    @test_throws ErrorException shared_genotype_case(true,"BayesC",5,123;prediction_rows=[2,1])
    @test_throws ErrorException shared_genotype_case(true,"BayesC",5,123;weights=fill(2.0,36))
    g = get_genotypes(Float32[0 1;1 2;2 0],1.0;quality_control=false,copy_genotypes=false)
    global shared_genotype_guard_genotypes = g
    model = build_model("y = intercept + shared_genotype_guard_genotypes",1.0)
    @test_throws ErrorException JWAS.validate_shared_genotypes(model;double_precision=true)
    @test_throws ErrorException JWAS.validate_shared_genotypes(model;single_step_analysis=true)
    @test_throws ErrorException JWAS.validate_shared_genotypes(model;RRM=ones(3,2))
    g.method = "BayesB"
    @test_throws ErrorException JWAS.validate_shared_genotypes(model)
end

@testset "Default sharing behavior" begin
    @test_throws ErrorException shared_genotype_case(true,"BayesC",5,123;
        selected_train=[1,3,5],shared_override=nothing)
    model = build_model("y = intercept",1.0)
    ph = DataFrame(ID=string.(1:8),y=Float64[1,3,2,4,2,5,1,4])
    mktempdir() do dir
        output = runMCMC(model,ph;chain_length=20,burnin=5,seed=73,
                         output_folder=joinpath(dir,"mcmc"),printout_model_info=false)
        @test haskey(output,"residual variance")
    end
end
