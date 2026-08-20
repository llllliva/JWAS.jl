using CSV
using DataFrames
using JWAS
using LinearAlgebra
using Random
using Statistics

BLAS.set_num_threads(1)

const N_OBSERVATIONS = 100
const N_MARKERS = 500
const N_ANNOTATIONS = 10
const CHAIN_LENGTH = 1_000
const BURNIN = 200
const BLOCK_STARTS = collect(1:100:N_MARKERS)
const SEEDS = [20260819, 20260820, 20260821]
const DEFAULT_OUTFILE = joinpath(
    @__DIR__,
    "reports",
    "2026-08-19-annotation-enrichment-overhead-results.csv",
)

function benchmark_inputs()
    rng = MersenneTwister(20260819)
    genotypes = Float32.(rand(rng, 0:2, N_OBSERVATIONS, N_MARKERS))
    annotations = Float64.(rand(rng, N_MARKERS, N_ANNOTATIONS) .< 0.2)
    annotations[1:N_ANNOTATIONS, :] .= Matrix{Float64}(I, N_ANNOTATIONS, N_ANNOTATIONS)
    phenotypes = DataFrame(
        ID=string.(1:N_OBSERVATIONS),
        y=randn(rng, N_OBSERVATIONS),
    )
    return genotypes, annotations, phenotypes
end

function build_case(genotypes, annotations)
    genotype_df = DataFrame(genotypes, Symbol.("m" .* string.(1:N_MARKERS)))
    insertcols!(genotype_df, 1, :ID => string.(1:N_OBSERVATIONS))
    genotype = get_genotypes(
        genotype_df,
        1.0;
        method="BayesC",
        Pi=0.2,
        estimatePi=true,
        annotations=annotations,
        quality_control=false,
    )
    @eval Main global annotation_enrichment_benchmark_geno = $genotype
    return build_model("y = intercept + annotation_enrichment_benchmark_geno", 1.0)
end

function run_case(genotypes, annotations, phenotypes, enabled, seed, output_folder; chain_length=CHAIN_LENGTH)
    model = build_case(genotypes, annotations)
    elapsed = @elapsed output = runMCMC(
        model,
        phenotypes;
        chain_length=chain_length,
        burnin=min(BURNIN, chain_length ÷ 5),
        output_samples_frequency=chain_length + 1,
        output_marker_effect_samples=false,
        outputEBV=false,
        output_heritability=false,
        output_annotation_enrichment=enabled,
        output_folder=output_folder,
        fast_blocks=BLOCK_STARTS,
        memory_guard=:off,
        printout_frequency=chain_length + 1,
        printout_model_info=false,
        seed=seed,
    )
    if enabled
        key = "annotation heritability enrichment annotation_enrichment_benchmark_geno"
        @assert haskey(output, key)
        @assert all(output[key].Posterior_Iterations .== chain_length - min(BURNIN, chain_length ÷ 5))
    end
    return elapsed
end

function main(args=ARGS)
    outfile = isempty(args) ? DEFAULT_OUTFILE : abspath(args[1])
    mkpath(dirname(outfile))
    genotypes, annotations, phenotypes = benchmark_inputs()
    rows = NamedTuple[]

    mktempdir() do tmpdir
        redirect_stdout(devnull) do
            run_case(genotypes, annotations, phenotypes, false, 1, joinpath(tmpdir, "warmup_off"); chain_length=5)
            run_case(genotypes, annotations, phenotypes, true, 1, joinpath(tmpdir, "warmup_on"); chain_length=5)
        end

        for (replicate, seed) in enumerate(SEEDS)
            order = isodd(replicate) ? (false, true) : (true, false)
            for enabled in order
                elapsed = redirect_stdout(devnull) do
                    run_case(
                        genotypes,
                        annotations,
                        phenotypes,
                        enabled,
                        seed,
                        joinpath(tmpdir, "rep$(replicate)_$(enabled ? "on" : "off")"),
                    )
                end
                push!(rows, (
                    replicate=replicate,
                    seed=seed,
                    enrichment_enabled=enabled,
                    elapsed_seconds=elapsed,
                    n_observations=N_OBSERVATIONS,
                    n_markers=N_MARKERS,
                    n_annotations=N_ANNOTATIONS,
                    chain_length=CHAIN_LENGTH,
                    burnin=BURNIN,
                    retained_outer_iterations=CHAIN_LENGTH - BURNIN,
                    block_count=length(BLOCK_STARTS),
                    julia_threads=Threads.nthreads(),
                    blas_threads=BLAS.get_num_threads(),
                ))
            end
        end
    end

    results = DataFrame(rows)
    CSV.write(outfile, results)
    off = results.elapsed_seconds[.!results.enrichment_enabled]
    on = results.elapsed_seconds[results.enrichment_enabled]
    println(results)
    println("median_off_seconds=", median(off))
    println("median_on_seconds=", median(on))
    println("median_overhead_percent=", 100 * (median(on) / median(off) - 1))
    paired_overhead = [
        100 * (
            only(results.elapsed_seconds[(results.replicate .== replicate) .& results.enrichment_enabled]) /
            only(results.elapsed_seconds[(results.replicate .== replicate) .& .!results.enrichment_enabled]) - 1
        ) for replicate in eachindex(SEEDS)
    ]
    println("paired_overhead_percent=", join(paired_overhead, ","))
    println("median_paired_overhead_percent=", median(paired_overhead))
    println("wrote=", outfile)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
