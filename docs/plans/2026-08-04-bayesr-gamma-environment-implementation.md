# BayesR gamma environment resolution implementation

## Implemented behavior

- Added `resolve_bayesr_gamma(gamma=nothing; env=ENV)`.
- Changed the `runMCMC` `BayesR_gamma` default to `nothing` so omission can be
  distinguished from an explicit keyword.
- Resolution precedence is explicit keyword, non-empty `BAYESR_GAMMA`
  environment value, then the existing package default.
- Environment values are parsed as four comma-separated floating-point values
  and pass through the existing BayesR gamma validation.
- Missing and whitespace-only environment values preserve the original
  `[0.0, 0.01, 0.1, 1.0]` default.
- Updated the API documentation and BayesR gamma diagnostic guide.

## Verification

- Focused runtime and MCMC tests: 36/36 passed.
- Full default unit suite: 1,044/1,044 passed; integration tests remained
  disabled by the repository's default configuration.
- Coverage includes default fallback, valid environment parsing, whitespace,
  malformed syntax, invalid ordering, explicit keyword precedence, and an
  MCMC run that resolves gamma only from the environment and writes it to the
  class-variance output.
- The documentation build completed source expansion, cross-reference, and
  document-check phases, then failed during HTML rendering in Documenter/Pkg
  with `project_rel_path(::Pkg.Types.EnvCache, ::Nothing)`. The same failure
  occurred with the normal and temporary depots and is unrelated to the edited
  Markdown content.
