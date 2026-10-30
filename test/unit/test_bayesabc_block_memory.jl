using Test
using JWAS
using Random
using LinearAlgebra

function make_block_inputs(nobs::Int,nmarkers::Int,block_size::Int)
    X = randn(Float64,nobs,nmarkers)
    Rinv = ones(Float64,nobs)
    starts = collect(1:block_size:nmarkers)
    XArray = [view(X,:,s:min(s+block_size-1,nmarkers)) for s in starts]
    XpRinvX = [Xb' * Xb for Xb in XArray]
    xpRinvx = [dot(view(X,:,j),view(X,:,j)) for j in 1:nmarkers]
    yCorr = randn(Float64,nobs)
    α = randn(Float64,nmarkers)
    β = copy(α)
    δ = ones(Float64,nmarkers)
    return (XArray,xpRinvx,X,XpRinvX,yCorr,α,β,δ,1.0,fill(1.0,nmarkers),0.95,Rinv)
end

@testset "BayesABC shared-row residual consistency" begin
    # Training rows occupy only part of the parent, so column stride differs
    # from the training sample size, as in the memory-saving production path.
    for T in (Float32, Float64), seed in (73, 123, 456)
        rng = MersenneTwister(seed)
        backing = randn(rng,T,41,11)
        original = copy(backing)
        X = view(backing,1:29,:)
        starts = 1:4:11
        blocks = [view(X,:,s:min(s+3,11)) for s in starts]
        grams = [b' * b for b in blocks]
        diagonals = T[dot(view(X,:,j),view(X,:,j)) for j in 1:11]
        residual = randn(rng,T,29)
        before = copy(residual)
        alpha = randn(rng,T,11)
        oldalpha = copy(alpha)
        beta = copy(alpha)
        delta = ones(T,11)
        Random.seed!(seed)
        JWAS.BayesABC_block!(blocks,diagonals,X,grams,residual,alpha,beta,delta,
                            one(T),ones(T,11),T(0.95),ones(T,29))
        expected = before + X*(oldalpha-alpha)
        @test isapprox(residual,expected;atol=100eps(T),rtol=100eps(T))
        @test eltype(residual) === T && eltype(alpha) === T
        @test backing == original
        @test all(isfinite,residual) && all(isfinite,alpha)
        @test all(d -> d == 0 || d == 1,delta)
    end
end

function measure_block_alloc(nobs::Int,nmarkers::Int,block_size::Int)
    args = make_block_inputs(nobs,nmarkers,block_size)
    Random.seed!(123)
    JWAS.BayesABC_block!(args...)
    args = make_block_inputs(nobs,nmarkers,block_size)
    Random.seed!(123)
    return @allocated JWAS.BayesABC_block!(args...)
end

@testset "BayesABC_block allocation scaling" begin
    nobs = 128
    block_size = 8
    alloc_small = measure_block_alloc(nobs,64,block_size)
    alloc_large = measure_block_alloc(nobs,512,block_size)
    @test alloc_large <= alloc_small * 12
end
