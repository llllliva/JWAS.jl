# Run this same script with --project pointing to either untouched upstream or
# the port. First argument is a new output directory. No new keywords are used.
using JWAS, DataFrames, CSV, Random, Statistics, Test
outdir = abspath(only(ARGS))
mkpath(outdir)
for method in ("BayesC","BayesR"), seed in (73,123), fast in (false,20)
    rng = MersenneTwister(20261021)
    n,t,p = 128,96,257
    X = Float32.(rand(rng,0:2,n,p))
    ids = ["animal_$(lpad(i,4,'0'))" for i in 1:n]
    markers = ["rs_$(1000+7*i)" for i in 1:p]
    observations = 0.3 .* Float64.(X[:,1]) .+ 0.1 .* collect(1:n)./n .+ randn(rng,n)
    pi = method == "BayesC" ? 0.65 : [0.65,0.25,0.08,0.02]
    global compat_genotypes = get_genotypes(X,1.0;method,Pi=pi,
        quality_control=false,double_precision=false)
    compat_genotypes.obsID .= ids
    compat_genotypes.markerID .= markers
    ph = DataFrame(ID=ids,y=Union{Missing,Float64}[i<=t ? observations[i] : missing for i in 1:n],
        FARM=ifelse.(iseven.(1:n),"A","B"),LITTER=["L$(mod1(i,6))" for i in 1:n])
    model = build_model("y = intercept + FARM + LITTER + compat_genotypes",1.0)
    set_random(model,"LITTER",1.0)
    outputEBV(model,ids)
    prefix = "$(method)_seed$(seed)_block$(fast)"
    folder = joinpath(outdir,prefix)
    result = runMCMC(model,ph;chain_length=(fast === false ? 300 : 300*fast),
        burnin=100,output_samples_frequency=10,fast_blocks=fast,seed,
        output_folder=folder,double_precision=false,output_heritability=false,
        printout_model_info=false,printout_frequency=301)
    animals = CSV.read(joinpath(folder,"EBV_y.txt"),DataFrame;types=Dict(:ID=>String))
    effects = result["marker effects compat_genotypes"]
    @test nrow(animals) == n && Set(String.(animals.ID)) == Set(ids)
    @test nrow(effects) == p && Set(string.(effects.Marker_ID)) == Set(markers)
    CSV.write(joinpath(outdir,prefix*"_animals.csv"),animals)
    CSV.write(joinpath(outdir,prefix*"_markers.csv"),effects)
end
println("Completed output export: ",outdir)
