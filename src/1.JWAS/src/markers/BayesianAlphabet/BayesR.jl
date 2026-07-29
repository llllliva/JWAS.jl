@inline function bayesr_logsumexp(log_probs)
    max_log = maximum(log_probs)
    return max_log + log(sum(exp.(log_probs .- max_log)))
end

@inline bayesr_prior_row(π::AbstractVector, ::Integer) = π
@inline bayesr_prior_row(π::AbstractMatrix, j::Integer) = view(π, j, :)

function bayesr_validate_priors(π::AbstractVector, nclasses::Integer, ::Integer)
    length(π) == nclasses || error("BayesR pi vector length $(length(π)) must match the number of mixture classes ($nclasses).")
    all(x -> x >= 0, π) || error("BayesR pi entries must be nonnegative.")
    isapprox(sum(π), 1.0; atol=1e-8) || error("BayesR pi must sum to 1.")
    return nothing
end

function bayesr_validate_priors(π::AbstractMatrix, nclasses::Integer, nmarkers::Integer)
    size(π, 1) == nmarkers || error("BayesR per-marker pi must have one row per marker.")
    size(π, 2) == nclasses || error("BayesR per-marker pi must have $nclasses columns.")
    return nothing
end

@inline function bayesr_block_nreps(iter::Integer, burnin::Integer, block_size::Integer)
    block_size < 1 && error("BayesR block_size must be at least 1.")
    return iter <= burnin ? 1 : Int(block_size)
end

mutable struct BayesRInnerOuterStats
    class_counts::Matrix{UInt64}
    effect_sum::Vector{Float64}
    effect_sum2::Vector{Float64}
    draws::Vector{UInt64}
end

function BayesRInnerOuterStats(nmarkers::Integer, nclasses::Integer)
    nmarkers > 0 || error("BayesR inner/outer diagnostics require at least one marker.")
    nclasses > 1 || error("BayesR inner/outer diagnostics require at least two classes.")
    return BayesRInnerOuterStats(
        zeros(UInt64, nmarkers, nclasses),
        zeros(Float64, nmarkers),
        zeros(Float64, nmarkers),
        zeros(UInt64, nmarkers),
    )
end

@inline function record_bayesr_inner_draw!(stats::BayesRInnerOuterStats,
                                           effect,
                                           sampled_class::Integer,
                                           marker::Integer)
    1 <= sampled_class <= size(stats.class_counts, 2) ||
        error("BayesR sampled class $sampled_class is outside the diagnostics class range.")
    effect64 = Float64(effect)
    stats.class_counts[marker, sampled_class] += 1
    stats.effect_sum[marker] += effect64
    stats.effect_sum2[marker] += effect64 * effect64
    stats.draws[marker] += 1
    return nothing
end

function record_bayesr_inner_sweep!(stats::BayesRInnerOuterStats,
                                    α,
                                    δ,
                                    marker_range)
    for marker in marker_range
        record_bayesr_inner_draw!(stats, α[marker], Int(δ[marker]), marker)
    end
    return nothing
end

function summarize_bayesr_inner_outer(stats::BayesRInnerOuterStats)
    nmarkers, nclasses = size(stats.class_counts)
    effect_mean = fill(NaN, nmarkers)
    effect_sd = fill(NaN, nmarkers)
    inclusion_pip = fill(NaN, nmarkers)
    mean_delta = fill(NaN, nmarkers)
    class_pip = fill(NaN, nmarkers, nclasses)

    for marker in 1:nmarkers
        ndraws = stats.draws[marker]
        ndraws == 0 && continue
        denom = Float64(ndraws)
        effect_mean[marker] = stats.effect_sum[marker] / denom
        effect_second_moment = stats.effect_sum2[marker] / denom
        effect_sd[marker] = sqrt(max(0.0, effect_second_moment - effect_mean[marker]^2))
        for classi in 1:nclasses
            class_pip[marker, classi] = stats.class_counts[marker, classi] / denom
        end
        inclusion_pip[marker] = 1.0 - class_pip[marker, 1]
        mean_delta[marker] = sum(
            classi * stats.class_counts[marker, classi] for classi in 1:nclasses
        ) / denom
    end

    return (
        draws=stats.draws,
        effect_mean=effect_mean,
        effect_sd=effect_sd,
        inclusion_pip=inclusion_pip,
        mean_delta=mean_delta,
        class_pip=class_pip,
    )
end

function BayesR!(genotypes, ycorr, vare)
    priors = genotypes.annotations === false ? genotypes.π : genotypes.annotations.snp_pi
    BayesR!(genotypes.mArray, genotypes.mRinvArray, genotypes.mpRinvm,
            ycorr, genotypes.α[1], genotypes.δ[1], vare, genotypes.G.val, priors, BAYESR_GAMMA)
end

function BayesR_block!(genotypes, ycorr, vare, Rinv=ones(eltype(ycorr), length(ycorr)), independent_blocks=false)
    BayesR_block!(genotypes, ycorr, vare, Rinv, typemax(Int), 0, independent_blocks)
end

function BayesR_block!(genotypes, ycorr, vare, Rinv, iter::Integer, burnin::Integer, independent_blocks=false)
    priors = genotypes.annotations === false ? genotypes.π : genotypes.annotations.snp_pi
    BayesR_block!(genotypes.MArray, genotypes.mpRinvm,
                  genotypes.genotypes, genotypes.MpRinvM,
                  ycorr, genotypes.α[1], genotypes.δ[1], vare,
                  genotypes.G.val, priors, BAYESR_GAMMA, Rinv, iter, burnin,
                  independent_blocks, genotypes.innerOuterStats)
end

function BayesR!(xArray, xRinvArray, xpRinvx,
                 yCorr,
                 α, δ,
                 vare, sigmaSq, π, gamma)
    nclasses = length(gamma)
    bayesr_validate_priors(π, nclasses, length(α))
    sigmaSq > 0 || error("BayesR sigmaSq must be positive.")

    log_probs = zeros(Float64, nclasses)
    probs = zeros(Float64, nclasses)
    invVarRes = 1 / vare

    for j in eachindex(α)
        x = xArray[j]
        xRinv = xRinvArray[j]
        rhs = (dot(xRinv, yCorr) + xpRinvx[j] * α[j]) * invVarRes
        oldAlpha = α[j]
        πj = bayesr_prior_row(π, j)

        log_probs[1] = log(πj[1])
        for k in 2:nclasses
            varEffect = gamma[k] * sigmaSq
            invVarEffect = 1 / varEffect
            lhs = xpRinvx[j] * invVarRes + invVarEffect
            invLhs = 1 / lhs
            betaHat = invLhs * rhs
            log_probs[k] = 0.5 * (log(invLhs) - log(varEffect) + betaHat * rhs) + log(πj[k])
        end

        log_norm = bayesr_logsumexp(log_probs)
        for k in 1:nclasses
            probs[k] = exp(log_probs[k] - log_norm)
        end

        sampled_class = rand(Categorical(probs))
        δ[j] = sampled_class

        if sampled_class == 1
            if oldAlpha != 0
                BLAS.axpy!(oldAlpha, x, yCorr)
            end
            α[j] = zero(eltype(α))
        else
            varEffect = gamma[sampled_class] * sigmaSq
            invVarEffect = 1 / varEffect
            lhs = xpRinvx[j] * invVarRes + invVarEffect
            invLhs = 1 / lhs
            betaHat = invLhs * rhs
            α[j] = betaHat + randn() * sqrt(invLhs)
            BLAS.axpy!(oldAlpha - α[j], x, yCorr)
        end
    end
end

function BayesR_block!(XArray, xpRinvx,
                       X, XpRinvX,
                       yCorr,
                       α, δ,
                       vare, sigmaSq, π, gamma, Rinv)
    BayesR_block!(XArray, xpRinvx,
                  X, XpRinvX,
                  yCorr,
                  α, δ,
                  vare, sigmaSq, π, gamma, Rinv, typemax(Int), 0, false)
end

function BayesR_block!(XArray, xpRinvx,
                       X, XpRinvX,
                       yCorr,
                       α, δ,
                       vare, sigmaSq, π, gamma, Rinv, iter::Integer, burnin::Integer,
                       independent_blocks=false, inner_outer_stats=false)
    if independent_blocks == true
        return BayesR_block_independent!(XArray, xpRinvx,
                                         X, XpRinvX,
                                         yCorr,
                                         α, δ,
                                         vare, sigmaSq, π, gamma, Rinv, iter, burnin,
                                         inner_outer_stats)
    end

    nclasses = length(gamma)
    bayesr_validate_priors(π, nclasses, length(α))
    sigmaSq > 0 || error("BayesR sigmaSq must be positive.")

    log_probs = zeros(Float64, nclasses)
    probs = zeros(Float64, nclasses)
    invVarRes = 1 / vare
    nblocks = length(XpRinvX)
    start_pos = 0
    max_block_size = maximum(size(XpRinvX[i], 1) for i in 1:nblocks)
    αold_block = Vector{eltype(α)}(undef, max_block_size)
    rhs_block = Vector{eltype(yCorr)}(undef, max_block_size)
    unit_weights = is_unit_weights(Rinv)

    for i in 1:nblocks
        block_size = size(XpRinvX[i], 1)
        block_range = start_pos + 1:start_pos + block_size
        copyto!(view(αold_block, 1:block_size), view(α, block_range))
        XpRinvycorr = view(rhs_block, 1:block_size)
        block_rhs!(XpRinvycorr, XArray[i], yCorr, Rinv, unit_weights)
        nreps = bayesr_block_nreps(iter, burnin, block_size)

        for reps in 1:nreps
            for j in 1:block_size
                locus_j = start_pos + j
                rhs = (XpRinvycorr[j] + xpRinvx[locus_j] * α[locus_j]) * invVarRes
                oldAlpha = α[locus_j]
                πj = bayesr_prior_row(π, locus_j)

                log_probs[1] = log(πj[1])
                for k in 2:nclasses
                    varEffect = gamma[k] * sigmaSq
                    invVarEffect = 1 / varEffect
                    lhs = xpRinvx[locus_j] * invVarRes + invVarEffect
                    invLhs = 1 / lhs
                    betaHat = invLhs * rhs
                    log_probs[k] = 0.5 * (log(invLhs) - log(varEffect) + betaHat * rhs) + log(πj[k])
                end

                log_norm = bayesr_logsumexp(log_probs)
                for k in 1:nclasses
                    probs[k] = exp(log_probs[k] - log_norm)
                end

                sampled_class = rand(Categorical(probs))
                δ[locus_j] = sampled_class

                if sampled_class == 1
                    α[locus_j] = zero(eltype(α))
                else
                    varEffect = gamma[sampled_class] * sigmaSq
                    invVarEffect = 1 / varEffect
                    lhs = xpRinvx[locus_j] * invVarRes + invVarEffect
                    invLhs = 1 / lhs
                    betaHat = invLhs * rhs
                    α[locus_j] = betaHat + randn() * sqrt(invLhs)
                end

                BLAS.axpy!(oldAlpha - α[locus_j], view(XpRinvX[i], :, j), XpRinvycorr)
                if inner_outer_stats !== false && iter > burnin
                    record_bayesr_inner_draw!(
                        inner_outer_stats,
                        α[locus_j],
                        sampled_class,
                        locus_j,
                    )
                end
            end
        end

        @views begin
            αold_view = view(αold_block, 1:block_size)
            αold_view .-= α[block_range]
            mul!(yCorr, XArray[i], αold_view, 1.0, 1.0)
        end
        start_pos += block_size
    end
end

function BayesR_block_independent!(XArray, xpRinvx,
                                   X, XpRinvX,
                                   yCorr,
                                   α, δ,
                                   vare, sigmaSq, π, gamma, Rinv, iter::Integer, burnin::Integer,
                                   inner_outer_stats=false)
    nclasses = length(gamma)
    bayesr_validate_priors(π, nclasses, length(α))
    sigmaSq > 0 || error("BayesR sigmaSq must be positive.")

    invVarRes = 1 / vare
    nblocks = length(XpRinvX)
    block_sizes = [size(XpRinvX[i], 1) for i in 1:nblocks]
    block_offsets = cumsum([0; block_sizes[1:end-1]])
    yCorr_snapshot = copy(yCorr)
    unit_weights = is_unit_weights(Rinv)
    block_deltas = Vector{Vector{eltype(α)}}(undef, nblocks)
    block_seeds = rand(UInt, nblocks)

    Threads.@threads for i in 1:nblocks
        rng = MersenneTwister(block_seeds[i])
        log_probs = zeros(Float64, nclasses)
        probs = zeros(Float64, nclasses)
        block_size = block_sizes[i]
        start_pos = block_offsets[i]
        block_range = start_pos + 1:start_pos + block_size
        old_alpha = copy(view(α, block_range))
        XpRinvycorr = Vector{eltype(yCorr)}(undef, block_size)
        block_rhs!(XpRinvycorr, XArray[i], yCorr_snapshot, Rinv, unit_weights)
        nreps = bayesr_block_nreps(iter, burnin, block_size)

        for reps in 1:nreps
            for j in 1:block_size
                locus_j = start_pos + j
                rhs = (XpRinvycorr[j] + xpRinvx[locus_j] * α[locus_j]) * invVarRes
                oldAlpha = α[locus_j]
                πj = bayesr_prior_row(π, locus_j)

                log_probs[1] = log(πj[1])
                for k in 2:nclasses
                    varEffect = gamma[k] * sigmaSq
                    invVarEffect = 1 / varEffect
                    lhs = xpRinvx[locus_j] * invVarRes + invVarEffect
                    invLhs = 1 / lhs
                    betaHat = invLhs * rhs
                    log_probs[k] = 0.5 * (log(invLhs) - log(varEffect) + betaHat * rhs) + log(πj[k])
                end

                log_norm = bayesr_logsumexp(log_probs)
                for k in 1:nclasses
                    probs[k] = exp(log_probs[k] - log_norm)
                end

                sampled_class = rand(rng, Categorical(probs))
                δ[locus_j] = sampled_class

                if sampled_class == 1
                    α[locus_j] = zero(eltype(α))
                else
                    varEffect = gamma[sampled_class] * sigmaSq
                    invVarEffect = 1 / varEffect
                    lhs = xpRinvx[locus_j] * invVarRes + invVarEffect
                    invLhs = 1 / lhs
                    betaHat = invLhs * rhs
                    α[locus_j] = betaHat + randn(rng) * sqrt(invLhs)
                end

                BLAS.axpy!(oldAlpha - α[locus_j], view(XpRinvX[i], :, j), XpRinvycorr)
                if inner_outer_stats !== false && iter > burnin
                    record_bayesr_inner_draw!(
                        inner_outer_stats,
                        α[locus_j],
                        sampled_class,
                        locus_j,
                    )
                end
            end
        end

        old_alpha .-= α[block_range]
        block_deltas[i] = old_alpha
    end

    for i in 1:nblocks
        mul!(yCorr, XArray[i], block_deltas[i], 1.0, 1.0)
    end
    return nothing
end
