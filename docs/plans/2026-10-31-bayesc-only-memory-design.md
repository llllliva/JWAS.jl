# BayesC-only memory and residual optimization

Base: latest fetched reworkhow master 038c6c8 (JWAS 2.4.0).
Scope: BayesC ordinary and fast-block, single- and multi-trait; include both
existing MT samplers and sequential/independent blocks. No BayesR sampler edits.
BayesC defaults to no-copy matrix input and contiguous shared genotype views.
Other methods retain upstream defaults and block storage.

Compatibility residual arithmetic is the default. Opt-in
fast_residual_updates=true enables precision-matched in-place BLAS residual
updates, which may change seeded trajectories through floating-point rounding.
For MT sequential compatibility mode, reuse block-sized alpha and residual
workspaces, keeping matrix product then vector addition in upstream order.

Preserve priors, annotation models, sampling schedules and prediction formulas.
Support one dense BayesC term with matching precision, contiguous unique rows
and unit weights; keep opt-outs for unsupported layouts/modes. File/DataFrame
ownership stays unchanged. Validate default and copied storage, partial trait
missingness, held-out predictions, sampler I/II, both precisions and block modes.
Compare keyed predictions directly with upstream across multiple seeds; test
both residual modes. Run full regressions and documentation. Keep master and
upstream unchanged until review.
