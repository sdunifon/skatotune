//! Our own macOS bundle identifier.
//!
//! nice-plug-xtask writes `CFBundleIdentifier = com.nice-plug.<package>` into
//! every bundle's Info.plist. Hosts and macOS use that identifier (preferences,
//! code-signing requirements, crash reports), so it should be ours. It must
//! contain only letters, digits, hyphens and periods, so underscores in the
//! package name become hyphens.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

/// Reverse-DNS prefix for every bundle we ship.
pub const BUNDLE_ID_PREFIX: &str = "com.skatotune";

/// The identifier a package's bundles get, e.g. `com.skatotune.skatotune`.
pub fn bundle_identifier(package: &str) -> String {
    format!("{BUNDLE_ID_PREFIX}.{}", package.replace('_', "-"))
}

/// Replace nice-plug's default identifier in `bundle`'s Info.plist (no-op if
/// it is already ours or the plist is missing).
pub fn set_bundle_identifier(bundle: &Path, package: &str) -> Result<()> {
    let plist = bundle.join("Contents").join("Info.plist");
    let Ok(text) = fs::read_to_string(&plist) else {
        return Ok(());
    };
    let default = format!("<string>com.nice-plug.{package}</string>");
    let ours = format!("<string>{}</string>", bundle_identifier(package));
    if text.contains(&default) {
        fs::write(&plist, text.replace(&default, &ours))
            .with_context(|| format!("Could not write {}", plist.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_are_valid_reverse_dns() {
        assert_eq!(bundle_identifier("skatotune"), "com.skatotune.skatotune");
        assert_eq!(bundle_identifier("skatotune"), "com.skatotune.skatotune");
        assert!(bundle_identifier("a_b_c")
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'));
    }
}
