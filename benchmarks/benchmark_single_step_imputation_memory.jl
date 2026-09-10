using JWAS
using LinearAlgebra
using Random
using SparseArrays

mode = length(ARGS) == 1 ? Symbol(ARGS[1]) : :optimized
mode in (:legacy,:optimized) || error("usage: benchmark_single_step_imputation_memory.jl legacy|optimized")

# A moderate-scale analogue of the production SSBR matrices. Run each mode in
# a fresh Julia process under /usr/bin/time so maximum RSS includes the same
# package/runtime baseline.
n_genotyped = 4_000
n_nongenotyped = 500
n_markers = 10_000
n_pedigree = n_genotyped+n_nongenotyped
n_output = 500

Random.seed!(20260909)
observed = rand(Float32,n_genotyped,n_markers)
Zgeno = spdiagm(0 => ones(Float64,n_genotyped))
lhs = spdiagm(0 => ones(Float64,n_pedigree))

imputation_rows = sparse(
    1:n_nongenotyped,
    mod1.(1:n_nongenotyped,n_genotyped),
    fill(0.5,n_nongenotyped),
    n_nongenotyped,
    n_genotyped,
)
rhs = [imputation_rows; spdiagm(0 => ones(Float64,n_genotyped))]
Zpheno = spdiagm(0 => ones(Float64,n_pedigree))
Zout = sparse(1:n_output,(n_pedigree-n_output+1):n_pedigree,ones(Float64,n_output),n_output,n_pedigree)

if mode === :optimized
    Mpheno,Mout = JWAS._impute_genotypes_in_chunks(
        observed,Zgeno,lhs,rhs,Zpheno,Zout;
        markerperchunk=1_000,
        show_progress=false,
    )
else
    Mg = Zgeno*observed
    Mpheno = zeros(n_pedigree,n_markers)
    Mout = zeros(n_output,n_markers)
    markerperchunk = 1_000
    for chunk_start in 1:markerperchunk:n_markers
        myrange = chunk_start:min(chunk_start+markerperchunk-1,n_markers)
        Mped_chunk = lhs\(rhs*Mg[:,myrange])
        Mpheno[:,myrange] = Zpheno*Mped_chunk
        Mout[:,myrange] = Zout*Mped_chunk
    end
    Mout = Float32.(Mout)
    Mpheno = Float32.(Mpheno)
end

@assert eltype(Mpheno) === Float32
@assert eltype(Mout) === Float32
println("mode=$mode checksum=$(sum(Mpheno)+sum(Mout))")
println("maxrss_bytes=$(Sys.maxrss())")
