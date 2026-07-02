{
  description = "keccak-batch - batched Keccak-256 with runtime-dispatched SIMD";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Exactly the toolchain pinned in rust-toolchain.toml (channel,
        # components, and the wasm32 target), so the flake and a bare
        # `rustup`-style build agree.
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "keccak-batch-dev";

          buildInputs = with pkgs; [
            rustToolchain
            cargo-nextest # test runner
            git
          ];

          shellHook = ''
            echo "keccak-batch dev shell — $(rustc --version)"
            echo "native:  cargo test    |    cargo clippy --all-targets --all-features"
            echo "wasm:    cargo build --target wasm32-unknown-unknown [+simd128]"
          '';

          RUST_BACKTRACE = "1";
        };

        packages.default = rustToolchain;
      });
}
