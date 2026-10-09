//! `cargo xtask`: nice-plug's bundler plus third-party notices (LIC-1) and our
//! own bundle identifier.
//!
//! ```text
//! cargo xtask bundle skatotune --release   # nice-plug bundle, then the notices
//!                                            # and LICENSE go into every bundle
//! cargo xtask notices                        # regenerate THIRD-PARTY-NOTICES.txt
//! cargo xtask notices --check                # fail if it is out of date
//! ```
//!
//! Everything else is passed to `nice_plug_xtask` unchanged.

mod identity;
mod notices;

fn main() -> nice_plug_xtask::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("notices") => {
            nice_plug_xtask::chdir_workspace_root()?;
            notices::command(&args[1..])
        }
        Some("bundle" | "bundle-universal") => {
            nice_plug_xtask::main_with_args("cargo xtask", args.iter().cloned())?;
            // nice-plug leaves the working directory at the workspace root.
            notices::install_into_bundles(&args[1..])
        }
        _ => nice_plug_xtask::main_with_args("cargo xtask", args),
    }
}
