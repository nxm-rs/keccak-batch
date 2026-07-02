# Releasing keccak-batch

This crate is released locally by a maintainer with [`cargo-release`](https://github.com/crate-ci/cargo-release) and [`git-cliff`](https://github.com/orhun/git-cliff). Cutting a release is a deliberate, signed, local action; CI does not publish. The configuration lives in `release.toml` (cargo-release) and `cliff.toml` (git-cliff). Pushing the release tag triggers `.github/workflows/release.yml`, which re-runs the checks and creates the GitHub release.

The project is licensed `AGPL-3.0-or-later`.

## Versioning

The crate is at `0.x`, so a `minor` bump is the normal release level and may carry breaking changes until `1.0`. Use `patch` for fix-only releases. Each release is tagged `vX.Y.Z`.

## Prerequisites

- A crates.io account that owns `keccak-batch`, and a token available to cargo: run `cargo login`, or export `CARGO_REGISTRY_TOKEN`.
- Your hardware signing key (YubiKey) unlocked in the gpg-agent, so cargo-release can create the signed release commit and signed `vX.Y.Z` tag without prompting.
- A clean checkout of `main` with no uncommitted changes, up to date with the remote.
- CI green on the commit you are about to release.
- `cargo-release` and `git-cliff` installed locally. On the Nix dev shell: `nix-shell -p cargo-release git-cliff`.

## Cutting a release

From a clean `main`:

```bash
# Dry run first: shows the version bump, changelog, publish plan, and tag.
cargo release minor

# When it looks right, execute it.
cargo release minor --execute
```

`cargo release minor --execute` will, in order:

1. Bump `version` in `Cargo.toml` to the next minor.
2. Run the pre-release hook (`git-cliff`) to regenerate `CHANGELOG.md` for the new version.
3. Create the signed release commit `chore(release): X.Y.Z`.
4. Create the signed tag `vX.Y.Z`.
5. Publish `keccak-batch` to crates.io.
6. Push the release commit and tag to the remote.

Use `patch` instead of `minor`, or `cargo release X.Y.Z --execute` to set an exact version. For the very first release the version in `Cargo.toml` is already the one to ship, so release it as-is:

```bash
cargo release --execute
```

## Post-release

- Verify the crate is live on crates.io and that docs.rs has built (`https://docs.rs/keccak-batch`).
- Check that the tag-triggered workflow created the GitHub release for `vX.Y.Z`.
