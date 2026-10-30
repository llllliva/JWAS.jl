# Run from repository root with julia --project=. benchmarks/upstream_genotype_storage.jl
using Test, JWAS, DataFrames, CSV, Random, Statistics
include(joinpath(@__DIR__, "..", "test", "unit", "test_shared_genotype_views.jl"))
report_dir = joinpath(@__DIR__, "reports", "2026-10-30-upstream-genotype-performance")
mkpath(report_dir)
metrics = NamedTuple[]
for method in ("BayesC","BayesR"), seed in (73,123), fast in (false,20)
    copied = shared_genotype_case(false,method,fast,seed;n=128,t=96,p=257)
    shared = shared_genotype_case(true,method,fast,seed;n=128,t=96,p=257)
    aligned = compare_shared_genotype_outputs(copied,shared)
    prefix = "$(method)_seed$(seed)_block$(fast)"
    CSV.write(joinpath(report_dir,prefix*"_animals.csv"),aligned.animals)
    CSV.write(joinpath(report_dir,prefix*"_markers.csv"),aligned.markers)
    push!(metrics,(;method,seed,block=fast,copied_bytes=copied.report.backing_matrix_bytes,
        shared_bytes=shared.report.backing_matrix_bytes,
        copied_parents=copied.report.backing_matrix_count,
        shared_parents=shared.report.backing_matrix_count,
        max_ebv_difference=maximum(abs.(aligned.animals.EBV-aligned.animals.EBV_1)),
        max_marker_difference=maximum(abs.(aligned.markers.Estimate-aligned.markers.Estimate_1))))
end
CSV.write(joinpath(report_dir,"storage.csv"),DataFrame(metrics))
println("Julia ",VERSION)
show(stdout,MIME("text/plain"),DataFrame(metrics)); println()
