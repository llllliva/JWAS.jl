#!/usr/bin/env julia
# Uses the production runMCMC path and explicit animal/SNP joins.
using LinearAlgebra
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"..","test","unit","test_pic_shared_matrix.jl"))
root = isempty(ARGS) ? joinpath(@__DIR__,"reports","2026-10-21-pic-shared-matrix") : abspath(only(ARGS))
mkpath(root)
metrics = DataFrame(method=String[],fast_block_size=Int[],seed=Int[],mode=String[],
    elapsed_seconds=Float64[],allocated_bytes=Int[],backing_matrix_count=Int[],
    backing_matrix_bytes=Int[],block_gram_bytes=Int[],max_abs_ebv_difference=Float64[],
    max_abs_marker_difference=Float64[])
for seed in (73,123), method in ("BayesC","BayesR"), fast in (false,50)
    tag = "$(method)_block$(Int(fast))_seed$(seed)"
    GC.gc(true)
    copied = pic_shared_case(false,method,fast,seed;n=1200,t=900,p=5001,outer=80,burnin=20,freq=5,
        folder=joinpath(root,tag*"_copied"))
    GC.gc(true)
    shared = pic_shared_case(true,method,fast,seed;n=1200,t=900,p=5001,outer=80,burnin=20,freq=5,
        folder=joinpath(root,tag*"_shared"))
    joined = compare_pic_shared_outputs(copied,shared)
    @assert first(joined.animals.ID) in copied.ebv.ID
    @assert first(joined.markers.Marker_ID) in string.(copied.marker.Marker_ID)
    CSV.write(joinpath(root,tag*"_animals_joined.csv"),joined.animals)
    CSV.write(joinpath(root,tag*"_markers_joined.csv"),joined.markers)
    ebv_error = maximum(abs.(joined.animals.EBV .- joined.animals.EBV_1))
    marker_error = maximum(abs.(joined.markers.Estimate .- joined.markers.Estimate_1))
    for (label,result) in [("copying_alignment",copied),("shared_views",shared)]
        r = result.report
        push!(metrics,(method,Int(fast),seed,label,result.timing.time,result.timing.bytes,
            r.backing_matrix_count,r.backing_matrix_bytes,r.block_gram_bytes,ebv_error,marker_error))
    end
    CSV.write(joinpath(root,"metrics.csv"),metrics)
    copied=nothing;shared=nothing;joined=nothing
    GC.gc(true)
end
println("Production-path benchmark metrics and keyed comparisons: ",root)
println("allocated_bytes is cumulative allocation during runMCMC, not peak RSS.")
