# Run from repository root: julia --project=. benchmarks/genotype_memory_residual.jl
using Test, JWAS, DataFrames, CSV, Random, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "..", "test", "unit", "test_shared_genotype_views.jl"))
report_dir = joinpath(@__DIR__, "reports", "2026-10-30-genotype-performance")
mkpath(report_dir)
metrics = NamedTuple[]
for seed in (73,123), fast in (false,20)
    copied = shared_genotype_case(false,"BayesC",fast,seed;n=128,t=96,p=257)
    shared = shared_genotype_case(true,"BayesC",fast,seed;n=128,t=96,p=257)
    aligned = compare_shared_genotype_outputs(copied,shared)
    prefix = "seed$(seed)_block$(fast)"
    CSV.write(joinpath(report_dir,prefix*"_animals.csv"),aligned.animals)
    CSV.write(joinpath(report_dir,prefix*"_markers.csv"),aligned.markers)
    push!(metrics,(;seed,block=fast,copied_bytes=copied.report.backing_matrix_bytes,
        shared_bytes=shared.report.backing_matrix_bytes,
        copied_parents=copied.report.backing_matrix_count,
        shared_parents=shared.report.backing_matrix_count,
        max_ebv_difference=maximum(abs.(aligned.animals.EBV-aligned.animals.EBV_1)),
        max_marker_difference=maximum(abs.(aligned.markers.Estimate-aligned.markers.Estimate_1))))
end
CSV.write(joinpath(report_dir,"storage.csv"),DataFrame(metrics))
# Residual kernel measurement, not whole-analysis runtime.
function residual_kernel!(y,X,d,a,repetitions)
    for _ in 1:repetitions
        mul!(y,X,d,a,a)
    end
end
kernel = NamedTuple[]
BLAS.set_num_threads(1)
for T in (Float32,Float64)
    rng = MersenneTwister(73)
    parent_X = randn(rng,T,4096,200)
    X = view(parent_X,1:3072,:)
    delta = randn(rng,T,200)
    y = zeros(T,3072)
    residual_kernel!(y,X,delta,1.0,2)
    residual_kernel!(y,X,delta,one(T),2)
    old = copy(y); new = copy(y)
    residual_kernel!(old,X,delta,1.0,1)
    residual_kernel!(new,X,delta,one(T),1)
    @test isapprox(old,new;atol=100eps(T),rtol=100eps(T))
    legacy = median([@elapsed residual_kernel!(y,X,delta,1.0,20) for _ in 1:7])/20
    matched = median([@elapsed residual_kernel!(y,X,delta,one(T),20) for _ in 1:7])/20
    push!(kernel,(;precision=string(T),legacy_seconds=legacy,matched_seconds=matched,ratio=legacy/matched))
end
CSV.write(joinpath(report_dir,"residual_kernel.csv"),DataFrame(kernel))
println("Julia ",VERSION,"; BLAS ",BLAS.get_config())
show(stdout,MIME("text/plain"),DataFrame(metrics)); println()
show(stdout,MIME("text/plain"),DataFrame(kernel)); println()
