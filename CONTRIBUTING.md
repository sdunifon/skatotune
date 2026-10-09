# Contributing to Skatotune

Thanks for helping. Skatotune is a research-heavy real-time audio project, so contributions are judged on measured results: a change should come with numbers from the evaluation harness, not only with "sounds better to me".

## Getting started

Requirements: Rust (the exact toolchain is pinned in `rust-toolchain.toml` and installs automatically), macOS on Apple Silicon for the primary target (other platforms build but are less tested).

```sh
cargo build --workspace
cargo test --workspace --release
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo xtask bundle skatotune --release      # VST3 + CLAP bundles in target/bundled/
cargo standalone                             # run the plugin without a DAW
```

The evaluation harness lives in `crates/skato_eval`:

```sh
cargo test -p skato_eval --release --test quality_gates   # the quality gates (tuning, settling, vibrato, latency, CPU)
cargo run --release -p skato_eval -- suite --out out      # synthetic singing suite → out/report.md
cargo run --release -p skato_eval -- latency              # measured vs reported latency
```

## Ground rules

- **Real-time safety.** Nothing on the audio thread may allocate, lock, log, make system calls or panic. Everything is allocated up front. The rules are in `docs/DESIGN.md` §2.1, and zero-allocation tests enforce them.
- **Latency is a contract.** The latency a mode reports to the host must equal the measured latency exactly; `skato_eval latency` checks every mode, voice range and sample rate.
- **Measure, don't guess.** DSP changes report before/after numbers from `skato_eval` (and the real-singing evaluation when it applies). Performance changes report criterion results and end-to-end engine numbers (`docs/PERFORMANCE.md`).
- **Never loosen a quality gate** to make a change pass. If a gate is wrong, open an issue with the evidence.
- **Licensing.** Only permissively licensed dependencies (MIT, Apache-2.0, ISC, BSD, Zlib and similar). Never copy code from GPL/LGPL/AGPL projects; reimplement from papers instead. Run `cargo xtask notices` after changing dependencies.

## Research

Design decisions are backed by research in `docs/` (`research.md`, `rust-performance.md`, and the topic reports under `docs/research/`). When you research something, read the existing reports first, write your findings back, and note which other features your findings could help. Techniques found for one feature often help another.

## Issues and commits

- Work is tracked in GitHub issues; reference them in commits ("Closes #12").
- Keep commits focused, with a message that explains what changed and why.
- Run the tests and clippy before pushing.

## AI assistance

Much of Skatotune was designed and written with AI coding assistants (Anthropic's Claude), under human direction and review. Contributions made with AI tools are welcome; you are responsible for understanding, testing and licensing what you submit.

## Conduct

Be kind and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
