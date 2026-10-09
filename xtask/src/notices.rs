//! Third-party notices for the shipped binaries (LIC-1).
//!
//! The plugin bundles and the standalone app statically link every Rust
//! package in `vocal_tune`'s normal-dependency closure. Most of them are
//! MIT/Apache-2.0/BSD/ISC/Zlib, which require their copyright and license
//! text to accompany binary distributions, and egui embeds fonts under the
//! SIL Open Font License 1.1, the Ubuntu Font Licence and the Hack / Bitstream
//! Vera licence, whose texts must travel with the fonts.
//!
//! [`generate`] walks that closure in `cargo metadata` (for both macOS
//! architectures, with the `standalone` feature) and writes one plain-text
//! file:
//!
//! - the packages, their versions and license expressions;
//! - the license elections for packages offering a copyleft alternative
//!   (self_cell: Apache-2.0 rather than GPL-2.0-only);
//! - every license text the packages ship (top-level `LICENSE*`, `COPYING*`,
//!   `NOTICE*`, ... files, the manifest's `license-file`, and the `.txt`
//!   files next to bundled fonts), identical texts merged with the list of
//!   packages they apply to. A package that ships no license file gets the
//!   standard text of its license with the authors from its manifest, marked
//!   as such.
//!
//! It refuses to produce a file (and so to bundle) when a shipped package has
//! no license at all, or only copyleft options, so a new dependency cannot
//! slip into the bundles unnoticed. Our own crates are ISC (`LICENSE`).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use cargo_metadata::{CargoOpt, DependencyKind, MetadataCommand, Package, PackageId};

/// The generated file, at the workspace root and in every bundle.
pub const NOTICES_FILE: &str = "THIRD-PARTY-NOTICES.txt";
/// The project's own license (ISC), copied into every bundle as `LICENSE.txt`.
const LICENSE_FILE: &str = "LICENSE";
/// Package whose shipped closure is listed when none is named.
const DEFAULT_PACKAGE: &str = "vocal_tune";
/// Optional features of the default package that ship (the standalone app).
const SHIPPED_FEATURES: &[&str] = &["standalone"];
/// Platforms the bundles are built for (a universal build contains both).
const PLATFORMS: &[&str] = &["aarch64-apple-darwin", "x86_64-apple-darwin"];
/// License files larger than this are not license files.
const MAX_TEXT_BYTES: u64 = 256 * 1024;

/// `cargo xtask notices [--check] [-p <package>]`.
pub fn command(args: &[String]) -> Result<()> {
    let check = args.iter().any(|a| a == "--check");
    let package = package_arg(args).unwrap_or_else(|| DEFAULT_PACKAGE.to_string());
    let text = generate(&package)?;
    let path = Path::new(NOTICES_FILE);
    if check {
        let current = fs::read_to_string(path).unwrap_or_default();
        if current != text {
            bail!("{NOTICES_FILE} is out of date: run `cargo xtask notices`");
        }
        eprintln!("{NOTICES_FILE} is up to date");
    } else {
        fs::write(path, &text).with_context(|| format!("Could not write {NOTICES_FILE}"))?;
        eprintln!("Wrote {NOTICES_FILE} ({} bytes)", text.len());
    }
    Ok(())
}

/// After `cargo xtask bundle <args>`: put `LICENSE.txt` and freshly
/// generated notices into `Contents/Resources` of every bundle of the
/// bundled packages, then re-sign the bundles (ad hoc, like nice-plug does)
/// so their sealed resources stay valid.
pub fn install_into_bundles(args: &[String]) -> Result<()> {
    let packages = bundle_packages(args);
    let license = fs::read_to_string(LICENSE_FILE).context("Could not read LICENSE")?;
    let target_dir = MetadataCommand::new()
        .manifest_path("Cargo.toml")
        .no_deps()
        .exec()
        .context("Could not run `cargo metadata`")?
        .target_directory
        .into_std_path_buf();
    for package in packages {
        let text = generate(&package)?;
        if fs::read_to_string(NOTICES_FILE).ok().as_deref() != Some(text.as_str()) {
            eprintln!(
                "WARNING: {NOTICES_FILE} in the repository is out of date; run `cargo xtask notices`"
            );
        }
        let name = bundle_name(&package);
        for ext in ["vst3", "clap", "app"] {
            let bundle = target_dir.join("bundled").join(format!("{name}.{ext}"));
            if !bundle.is_dir() {
                continue;
            }
            let resources = bundle.join("Contents").join("Resources");
            fs::create_dir_all(&resources)
                .with_context(|| format!("Could not create {}", resources.display()))?;
            fs::write(resources.join(NOTICES_FILE), &text)?;
            fs::write(resources.join("LICENSE.txt"), &license)?;
            crate::identity::set_bundle_identifier(&bundle, &package)?;
            sign(&bundle);
            eprintln!("Added license notices to '{}'", bundle.display());
        }
    }
    Ok(())
}

/// Ad hoc re-signing after adding resources (macOS only; a warning
/// otherwise, exactly like nice-plug's own signing step).
fn sign(bundle: &Path) {
    if !cfg!(target_os = "macos") {
        return;
    }
    let signed = Command::new("codesign")
        .args(["-f", "-s", "-"])
        .arg(bundle)
        .status()
        .is_ok_and(|s| s.success());
    if !signed {
        eprintln!("WARNING: could not re-sign '{}'", bundle.display());
    }
}

/// `-p <package>` / `--package <package>` from the arguments.
fn package_arg(args: &[String]) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == "-p" || w[0] == "--package")
        .map(|w| w[1].clone())
}

/// The packages `cargo xtask bundle` was asked to bundle: every `-p <name>`,
/// or else the first argument (nice-plug's own rule).
fn bundle_packages(args: &[String]) -> Vec<String> {
    let named: Vec<String> = args
        .windows(2)
        .filter(|w| w[0] == "-p" || w[0] == "--package")
        .map(|w| w[1].clone())
        .collect();
    if !named.is_empty() {
        return named;
    }
    match args.first() {
        Some(first) if !first.starts_with('-') => vec![first.clone()],
        _ => vec![DEFAULT_PACKAGE.to_string()],
    }
}

/// Display name of a package's bundles: `name` from its `bundler.toml`
/// section, else the package name (nice-plug's rule). A two-line parser is
/// enough for this file.
fn bundle_name(package: &str) -> String {
    let config = fs::read_to_string("bundler.toml").unwrap_or_default();
    let mut in_section = false;
    for line in config.lines().map(str::trim) {
        if line.starts_with('[') {
            in_section = line == format!("[{package}]");
        } else if in_section
            && let Some(value) = line.strip_prefix("name")
            && let Some(value) = value.trim_start().strip_prefix('=')
        {
            return value.trim().trim_matches('"').to_string();
        }
    }
    package.to_string()
}

/// The external packages linked into `package`'s binaries, by id.
fn shipped_packages(package: &str) -> Result<BTreeMap<PackageId, Package>> {
    let mut shipped = BTreeMap::new();
    for platform in PLATFORMS {
        let mut cmd = MetadataCommand::new();
        cmd.manifest_path("Cargo.toml").other_options(vec![
            "--filter-platform".to_string(),
            platform.to_string(),
            "--locked".to_string(),
        ]);
        if package == DEFAULT_PACKAGE {
            cmd.features(CargoOpt::SomeFeatures(
                SHIPPED_FEATURES
                    .iter()
                    .map(|f| format!("{package}/{f}"))
                    .collect(),
            ));
        }
        let meta = cmd.exec().context("Could not run `cargo metadata`")?;
        let resolve = meta.resolve.as_ref().context("No dependency graph")?;
        let workspace: BTreeSet<&PackageId> = meta.workspace_members.iter().collect();
        let root = meta
            .packages
            .iter()
            .find(|p| p.name.as_str() == package && workspace.contains(&p.id))
            .with_context(|| format!("No workspace package named '{package}'"))?;
        let nodes: HashMap<&PackageId, _> = resolve.nodes.iter().map(|n| (&n.id, n)).collect();
        let by_id: HashMap<&PackageId, &Package> =
            meta.packages.iter().map(|p| (&p.id, p)).collect();

        // Normal (linked) dependencies only: build scripts and dev-only
        // crates do not end up in the binaries.
        let mut stack = vec![&root.id];
        let mut seen = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let Some(node) = nodes.get(id) else { continue };
            for dep in &node.deps {
                if dep
                    .dep_kinds
                    .iter()
                    .any(|k| k.kind == DependencyKind::Normal)
                {
                    stack.push(&dep.pkg);
                }
            }
        }
        for id in seen {
            if !workspace.contains(id)
                && let Some(p) = by_id.get(id)
            {
                shipped.insert(id.clone(), (*p).clone());
            }
        }
    }
    Ok(shipped)
}

/// One license text and the packages it applies to.
struct TextEntry {
    label: String,
    body: String,
    packages: BTreeSet<String>,
}

/// The full notices file for `package`.
pub fn generate(package: &str) -> Result<String> {
    let shipped = shipped_packages(package)?;
    let mut packages: Vec<&Package> = shipped.values().collect();
    packages.sort_by(|a, b| (a.name.as_str(), &a.version).cmp(&(b.name.as_str(), &b.version)));

    let mut problems = Vec::new();
    let mut elections = Vec::new();
    // Normalized text → entry, in order of first appearance.
    let mut texts: Vec<TextEntry> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for p in &packages {
        let who = format!("{} {}", p.name.as_str(), p.version);
        let expr = p.license.clone().unwrap_or_default();
        if expr.is_empty() && p.license_file.is_none() {
            problems.push(format!("{who}: no license"));
        }
        if !expr.is_empty() {
            match permissive_choice(&expr) {
                Some(choice) if choice != expr && is_copyleft(&expr) => {
                    elections.push(format!("{who} ({expr}): used under {choice}."));
                }
                Some(_) => {}
                None => problems.push(format!("{who}: only copyleft options ({expr})")),
            }
        }
        let mut entries = license_texts(p);
        if is_copyleft(&expr) && permissive_choice(&expr).is_some() {
            // Elected the permissive license: the copyleft text the package
            // also ships (self_cell's LICENSE-GPLv2) does not apply.
            entries.retain(|(label, _, _)| !label.to_ascii_lowercase().contains("gpl"));
        }
        if !entries.iter().any(|(_, _, own)| *own) {
            match fallback_text(p, &expr) {
                Some(t) => entries.insert(0, t),
                None => problems.push(format!(
                    "{who}: ships no license file and '{expr}' has no built-in text"
                )),
            }
        }
        for (label, body, _) in entries {
            let key = normalize(&body);
            let i = *index.entry(key).or_insert_with(|| {
                texts.push(TextEntry {
                    label,
                    body,
                    packages: BTreeSet::new(),
                });
                texts.len() - 1
            });
            texts[i].packages.insert(who.clone());
        }
    }
    if !problems.is_empty() {
        bail!(
            "License problems in the shipped dependencies:\n  {}",
            problems.join("\n  ")
        );
    }

    let mut out = String::new();
    let title = "THIRD-PARTY SOFTWARE NOTICES FOR VOCAL TUNE";
    let _ = writeln!(out, "{title}\n{}\n", "=".repeat(title.len()));
    let features = if package == DEFAULT_PACKAGE {
        format!(" (with the `{}` feature)", SHIPPED_FEATURES.join("`, `"))
    } else {
        String::new()
    };
    let _ = writeln!(
        out,
        "Vocal Tune is licensed under the ISC license (LICENSE / LICENSE.txt).\n\
         Its VST3 and CLAP plugins and its standalone application also contain the\n\
         third-party software listed below, each used under the license shown.\n\
         The license texts follow the list; each names the packages it applies to.\n\
         \n\
         Generated by `cargo xtask notices` from the Cargo dependency graph of\n\
         `{package}`{features} for {}.\n\
         Do not edit by hand.\n",
        PLATFORMS.join(" and ")
    );

    let _ = writeln!(out, "License choices\n---------------\n");
    let _ = writeln!(
        out,
        "Where a package offers a choice of licenses (\"A OR B\"), Vocal Tune uses it\n\
         under a permissive one, and the texts the package ships are reproduced\n\
         below. Choices that include a copyleft license:\n"
    );
    if elections.is_empty() {
        let _ = writeln!(out, "(none)");
    }
    for e in &elections {
        let _ = writeln!(out, "- {e}");
    }
    let _ = writeln!(out);

    let heading = format!("Packages ({})", packages.len());
    let _ = writeln!(out, "{heading}\n{}\n", "-".repeat(heading.len()));
    for p in &packages {
        let expr = p.license.as_deref().unwrap_or("see license file");
        let repo = p.repository.as_deref().unwrap_or("");
        let _ = writeln!(out, "{} {}  [{expr}]  {repo}", p.name.as_str(), p.version);
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "License texts\n-------------");
    for (i, t) in texts.iter().enumerate() {
        let applies: Vec<&str> = t.packages.iter().map(String::as_str).collect();
        let _ = writeln!(
            out,
            "\n{}\n[{}] {}\nApplies to: {}\n\n{}",
            "~".repeat(72),
            i + 1,
            t.label,
            applies.join(", "),
            t.body.trim_end()
        );
    }
    // Trailing whitespace on each line is never meaningful here and makes
    // the file churn between generators.
    let mut clean: String = out
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    clean.push('\n');
    Ok(clean)
}

/// Whitespace-insensitive key for merging identical license texts.
fn normalize(body: &str) -> String {
    body.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// License texts a package ships: `(label, text, is_the_package's_own)`.
/// Own texts are top-level license files and the manifest's
/// `license-file`; texts next to bundled font files are extra.
fn license_texts(p: &Package) -> Vec<(String, String, bool)> {
    let Some(dir) = p
        .manifest_path
        .parent()
        .map(|d| d.as_std_path().to_path_buf())
    else {
        return Vec::new();
    };
    let mut found: Vec<(String, PathBuf, bool)> = Vec::new();
    for path in sorted_files(&dir) {
        let name = file_name(&path).to_ascii_lowercase();
        let is_license = [
            "license",
            "licence",
            "copying",
            "copyright",
            "notice",
            "unlicense",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix));
        if is_license {
            found.push((file_name(&path), path, true));
        }
    }
    if let Some(lf) = &p.license_file {
        let path = dir.join(lf.as_std_path());
        if !found.iter().any(|(_, f, _)| *f == path) {
            found.push((lf.to_string(), path, true));
        }
    }
    // Font licences: the .txt files next to bundled fonts (egui's fonts
    // ship OFL.txt, UFL.txt and the Hack licence there).
    for sub in sorted_dirs(&dir) {
        let files = sorted_files(&sub);
        let has_fonts = files.iter().any(|f| {
            let n = file_name(f).to_ascii_lowercase();
            [".ttf", ".otf", ".woff", ".woff2"]
                .iter()
                .any(|e| n.ends_with(e))
        });
        if has_fonts {
            for f in files {
                if file_name(&f).to_ascii_lowercase().ends_with(".txt") {
                    let label = format!("{}/{}", file_name(&sub), file_name(&f));
                    found.push((label, f, false));
                }
            }
        }
    }
    found
        .into_iter()
        .filter(|(_, path, _)| {
            fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() <= MAX_TEXT_BYTES)
        })
        .filter_map(|(label, path, own)| {
            let bytes = fs::read(&path).ok()?;
            let body = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
            (!body.trim().is_empty()).then_some((label, body, own))
        })
        .collect()
}

fn sorted_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn sorted_dirs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Copyleft license families (an `OR` choice including them is recorded).
fn is_copyleft(expr: &str) -> bool {
    ["GPL", "AGPL", "LGPL", "EUPL", "OSL", "CDDL", "EPL"]
        .iter()
        .any(|id| expr.contains(id))
}

/// Split `expr` on its top-level `OR`s.
fn alternatives(expr: &str) -> Vec<String> {
    let normalized = expr.replace(" / ", " OR ").replace('/', " OR ");
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    let tokens: Vec<&str> = normalized
        .split_whitespace()
        .flat_map(|t| split_parens(t))
        .collect();
    for tok in tokens {
        match tok {
            "(" => {
                depth += 1;
                current.push_str("( ");
            }
            ")" => {
                depth -= 1;
                current.push_str(") ");
            }
            "OR" if depth == 0 => out.push(std::mem::take(&mut current)),
            t => {
                current.push_str(t);
                current.push(' ');
            }
        }
    }
    out.push(current);
    out.into_iter()
        .map(|a| a.replace("( ", "(").replace(" )", ")").trim().to_string())
        .filter(|a| !a.is_empty())
        .collect()
}

fn split_parens(token: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = token;
    while let Some(r) = rest.strip_prefix('(') {
        out.push("(");
        rest = r;
    }
    let mut closing = 0;
    while let Some(r) = rest.strip_suffix(')') {
        closing += 1;
        rest = r;
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out.extend(std::iter::repeat_n(")", closing));
    out
}

/// The first alternative of `expr` without a copyleft license.
fn permissive_choice(expr: &str) -> Option<String> {
    alternatives(expr).into_iter().find(|a| !is_copyleft(a))
}

/// Standard texts for packages that ship no license file. Only licenses
/// whose standard text needs nothing but a copyright line.
const MIT_TEXT: &str = include_str!("../licenses/MIT.txt");
const ISC_TEXT: &str = include_str!("../licenses/ISC.txt");

/// The standard text of `expr`'s first license we have a template for, with
/// the package authors from its manifest as the copyright holders.
fn fallback_text(p: &Package, expr: &str) -> Option<(String, String, bool)> {
    let ids: Vec<&str> = expr
        .split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == '/')
        .filter(|t| !t.is_empty() && *t != "OR" && *t != "AND")
        .collect();
    let (id, template) = ["MIT", "ISC"]
        .into_iter()
        .find(|id| ids.contains(id))
        .map(|id| (id, if id == "MIT" { MIT_TEXT } else { ISC_TEXT }))?;
    let names: Vec<String> = p
        .authors
        .iter()
        .map(|a| match a.find('<') {
            Some(i) => a[..i].trim().to_string(),
            None => a.trim().to_string(),
        })
        .filter(|a| !a.is_empty())
        .collect();
    let holders = if names.is_empty() {
        format!("the {} authors", p.name.as_str())
    } else {
        names.join(", ")
    };
    let label = format!(
        "{id} (standard text: the package ships no license file; copyright holders from its manifest)"
    );
    Some((label, template.replace("{holders}", &holders), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn license_choices_are_split_at_the_top_level() {
        assert_eq!(alternatives("MIT OR Apache-2.0"), ["MIT", "Apache-2.0"]);
        assert_eq!(alternatives("MIT/Apache-2.0"), ["MIT", "Apache-2.0"]);
        assert_eq!(
            alternatives("(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0"),
            ["(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0"]
        );
        assert_eq!(
            permissive_choice("Apache-2.0 OR GPL-2.0-only").as_deref(),
            Some("Apache-2.0")
        );
        assert_eq!(
            permissive_choice("GPL-2.0-only OR Apache-2.0").as_deref(),
            Some("Apache-2.0")
        );
        assert_eq!(permissive_choice("GPL-3.0-only"), None);
        assert_eq!(permissive_choice("LGPL-2.1-or-later"), None);
    }

    #[test]
    fn bundle_arguments_name_the_packages() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            bundle_packages(&args(&["vocal_tune", "--release"])),
            ["vocal_tune"]
        );
        assert_eq!(
            bundle_packages(&args(&["-p", "a", "-p", "b", "--release"])),
            ["a", "b"]
        );
        assert_eq!(bundle_packages(&args(&["--release"])), [DEFAULT_PACKAGE]);
    }

    #[test]
    fn texts_merge_ignoring_whitespace() {
        assert_eq!(normalize("a  b\r\n c\n"), normalize("a b c"));
        assert_ne!(normalize("Copyright 2020 A"), normalize("Copyright 2020 B"));
    }
}
