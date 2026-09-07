using JWAS
using DataFrames
using Random
using Statistics

const NOBS = 100
const NMARKERS = 5_000
const BLOCK_SIZE = 10
const CHAIN_LENGTH = 300
const BURNIN = 50
const SEEDS = (20260907, 20260908, 20260909)

function benchmark_inputs(seed)
    rng = MersenneTwister(seed)
    genotypes = Float32.(rand(rng, 0:2, NOBS, NMARKERS))
    annotations = zeros(Float64, NMARKERS, 4)
    for marker in 1:NMARKERS
        annotations[marker, 1] = isodd(marker)
        annotations[marker, 2] = marker % 3 == 0
        annotations[marker, 3] = marker % 5 <= 1
        annotations[marker, 4] = marker % 11 == 0
    end
    effects = zeros(Float64, NMARKERS)
    effects[1:25] .= randn(rng, 25) .* 0.15
    phenotypes = DataFrame(
        ID=string.(1:NOBS),
        y=Float64.(genotypes) * effects + randn(rng, NOBS),
    )
    return genotypes, annotations, phenotypes
end

function benchmark_run(seed; diagnostics)
    matrix, annotations, phenotypes = benchmark_inputs(seed)
    global benchmark_bayesc_innerouter_geno = get_genotypes(
        matrix,
        1.0;
        method="BayesC",
        Pi=0.95,
        estimatePi=true,
        quality_control=false,
        copy_genotypes=false,
        annotations=annotations,
    )
    model = build_model("y = intercept + benchmark_bayesc_innerouter_geno", 1.0)
    block_starts = collect(1:BLOCK_SIZE:NMARKERS)

    result = mktempdir() do output_folder
        timed = @timed runMCMC(
            model,
            phenotypes;
            chain_length=CHAIN_LENGTH,
            burnin=BURNIN,
            output_samples_frequency=20,
            output_folder=output_folder,
            outputEBV=false,
            output_heritability=false,
            output_annotation_enrichment=false,
            output_marker_effect_samples=false,
            printout_model_info=false,
            printout_frequency=CHAIN_LENGTH + 1,
            fast_blocks=block_starts,
            inner_outer_diagnostics=diagnostics,
            memory_guard=:off,
            seed=seed,
        )
        marker_output = timed.value["marker effects benchmark_bayesc_innerouter_geno"]
        diagnostics && @assert all(marker_output.InnerOuter_Samples .==
                                   UInt64((CHAIN_LENGTH - BURNIN) * BLOCK_SIZE))
        return (seconds=timed.time, bytes=timed.bytes)
    end
    return result
end

# Compile both paths before the paired measurements.
benchmark_run(first(SEEDS); diagnostics=false)
benchmark_run(first(SEEDS); diagnostics=true)

disabled = NamedTuple[]
enabled = NamedTuple[]
for seed in SEEDS
    push!(disabled, benchmark_run(seed; diagnostics=false))
    push!(enabled, benchmark_run(seed; diagnostics=true))
end

disabled_seconds = getfield.(disabled, :seconds)
enabled_seconds = getfield.(enabled, :seconds)
paired_overhead = (enabled_seconds .- disabled_seconds) ./ disabled_seconds

println("nobs=", NOBS)
println("nmarkers=", NMARKERS)
println("block_size=", BLOCK_SIZE)
println("retained_outer_iterations=", CHAIN_LENGTH - BURNIN)
println("inner_draws_per_marker=", (CHAIN_LENGTH - BURNIN) * BLOCK_SIZE)
println("diagnostic_state_bytes_per_marker=40")
println("diagnostic_state_bytes=", 40 * NMARKERS)
for (index, seed) in enumerate(SEEDS)
    println(
        "seed=", seed,
        " disabled_seconds=", disabled_seconds[index],
        " enabled_seconds=", enabled_seconds[index],
        " paired_overhead=", paired_overhead[index],
    )
end
println("median_disabled_seconds=", median(disabled_seconds))
println("median_enabled_seconds=", median(enabled_seconds))
println("median_paired_overhead=", median(paired_overhead))
