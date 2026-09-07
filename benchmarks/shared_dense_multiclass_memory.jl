using JWAS
using DataFrames
using Random

Random.seed!(20260906)
nobs = parse(Int, get(ENV, "JWAS_SHARED_BENCH_NOBS", "100"))
nmarkers = parse(Int, get(ENV, "JWAS_SHARED_BENCH_NMARKERS", "1000"))

matrix = Float32.(rand(0:2, nobs, nmarkers))
memberships = falses(nmarkers, 3)
for marker in 1:nmarkers
    memberships[marker, 1] = marker % 2 == 0
    memberships[marker, 2] = marker % 3 != 0
    memberships[marker, 3] = marker % 5 <= 1
end

base = get_genotypes(
    matrix,
    1.0;
    method="BayesC",
    Pi=0.9,
    quality_control=false,
    center=true,
    copy_genotypes=false,
)

get_shared_genotype_classes(base, memberships; add_unassigned=true, Pi=0.9)
GC.gc()
construction_bytes = @allocated begin
    global shared_classes = get_shared_genotype_classes(
        base,
        memberships;
        add_unassigned=true,
        Pi=0.9,
    )
end

logical_class_values = sum(length(class.genotypes) for class in shared_classes)
base_bytes = sizeof(eltype(base.genotypes)) * length(base.genotypes)
duplicated_class_bytes = sizeof(eltype(base.genotypes)) * logical_class_values

global benchmark_class_1 = shared_classes[1]
global benchmark_class_2 = shared_classes[2]
global benchmark_class_3 = shared_classes[3]
model = build_model(
    "y = intercept + benchmark_class_1 + benchmark_class_2 + benchmark_class_3",
    1.0,
)
for Mi in model.M
    Mi.α = [randn(Float32, Mi.nMarkers)]
end
JWAS.multiclass_bayesc_class_heritability(model, 1.0)
class_h2_bytes = @allocated JWAS.multiclass_bayesc_class_heritability(model, 1.0)
for Mi in model.M
    Mi.α = false
end
phenotypes = DataFrame(ID=string.(1:nobs), y=randn(nobs))
mcmc_seconds = mktempdir() do output_folder
    @elapsed runMCMC(
        model,
        phenotypes;
        chain_length=2,
        burnin=0,
        output_samples_frequency=2,
        output_folder=output_folder,
        outputEBV=false,
        output_heritability=true,
        output_marker_effect_samples=false,
        printout_frequency=3,
        printout_model_info=false,
        memory_guard=:off,
        seed=906,
    )
end

println("nobs=", nobs)
println("nmarkers=", nmarkers)
println("memberships=", sum(class.nMarkers for class in shared_classes))
println("base_genotype_bytes=", base_bytes)
println("logical_duplicated_class_bytes=", duplicated_class_bytes)
println("shared_class_construction_allocated_bytes=", construction_bytes)
println("all_classes_alias_base=", all(class -> Base.mightalias(class.genotypes, base.genotypes), shared_classes))
println("class_h2_allocated_bytes=", class_h2_bytes)
println("two_iteration_mcmc_seconds=", mcmc_seconds)
