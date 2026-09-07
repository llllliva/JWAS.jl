using JWAS
using Random

Random.seed!(20260906)

nobs = 20
nmarkers = 100_000
nannotations = 20

genotype_matrix = Float32.(rand(0:2, nobs, nmarkers))
annotation_matrix = Float64.(rand(nmarkers, nannotations) .< 0.08)

# Guarantee that every annotation is present and retain some deliberately
# unannotated and overlapping markers.
for annotation in 1:nannotations
    annotation_matrix[annotation, annotation] = 1.0
end
annotation_matrix[1_000:1_100, :] .= 0.0
annotation_matrix[2_000:2_100, 1:4] .= 1.0

genotype = get_genotypes(
    genotype_matrix,
    1.0;
    method="BayesC",
    Pi=0.95,
    estimatePi=true,
    annotations=annotation_matrix,
    quality_control=false,
    center=false,
    copy_genotypes=false,
)
genotype.ntraits = 1
genotype.trait_names = ["trait"]
JWAS.finalize_marker_annotation_setup!(genotype)
genotype.α = [randn(Float32, nmarkers)]
genotype.annotations.coefficients[2:end] .= range(-1.0, 1.0; length=nannotations)
genotype.annotations.mu .= genotype.annotations.design_matrix * genotype.annotations.coefficients
genotype.annotations.enrichment_stats = JWAS.initialize_annotation_enrichment(genotype)

# Compile before measuring the production accumulation path.
JWAS.update_annotation_enrichment!(genotype, 1.0)
genotype.annotations.enrichment_stats = JWAS.initialize_annotation_enrichment(genotype)

elapsed = @elapsed JWAS.update_annotation_enrichment!(genotype, 1.0)
allocated = @allocated JWAS.update_annotation_enrichment!(genotype, 1.0)
result = JWAS.annotation_enrichment_dataframe(genotype)

println("nmarkers=$nmarkers")
println("nannotations=$nannotations")
println("memberships=$(Int(sum(annotation_matrix)))")
println("elapsed_seconds=$elapsed")
println("allocated_bytes=$allocated")
println("reported_groups=$(size(result, 1))")
println("proportion_sum=$(sum(result.H2_Proportion_Estimate))")
println("weighted_enrichment_sum=$(sum(result.SNP_Fraction .* result.Enrichment_Estimate))")
