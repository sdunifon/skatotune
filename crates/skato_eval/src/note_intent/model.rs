//! Convert a trained candidate-note tree ensemble from the training export
//! (`LDMODEL 1` text, `kind cand_gbt`, written by the training script) to the
//! compact `SKGB` binary the plugin embeds
//! (`skato_control::intent::gbt`). The text is
//! `key value` lines (`kind`, `nf`, `nb`, `hidden`) and `name count v1 v2 …`
//! arrays: `gbt_base`, `tree_sizes`, `node_feat` (−1 for a leaf),
//! `node_thr`, `node_left`, `node_right` (relative to the tree's first
//! node) and `node_val`. Values are parsed straight to `f32`, so the binary
//! holds exactly the numbers the prototype evaluated.

use skato_control::intent::gbt::{self, ExportNode};

/// Parse an `LDMODEL 1` candidate tree ensemble and encode it as `SKGB`.
pub fn convert_ldm(text: &str) -> Result<Vec<u8>, String> {
    let mut it = text.split_ascii_whitespace();
    if it.next() != Some("LDMODEL") || it.next() != Some("1") {
        return Err("not an LDMODEL 1 file".into());
    }
    let mut kind = None;
    let mut nf = 0usize;
    let mut arrays: Vec<(String, Vec<f32>)> = Vec::new();
    while let Some(key) = it.next() {
        match key {
            "kind" => kind = it.next().map(str::to_string),
            "nf" | "nb" | "hidden" => {
                let v: usize = it.next().and_then(|x| x.parse().ok()).ok_or_else(|| format!("bad value for {key}"))?;
                if key == "nf" {
                    nf = v;
                }
            }
            name => {
                let n: usize = it.next().and_then(|x| x.parse().ok()).ok_or_else(|| format!("bad count for {name}"))?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(it.next().and_then(|x| x.parse::<f32>().ok()).ok_or_else(|| format!("bad value in {name}"))?);
                }
                arrays.push((name.to_string(), v));
            }
        }
    }
    if kind.as_deref() != Some("cand_gbt") {
        return Err(format!("kind {kind:?}: only cand_gbt models can be converted"));
    }
    let take = |name: &str| arrays.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_slice()).ok_or_else(|| format!("missing {name}"));
    let base = *take("gbt_base")?.first().ok_or("empty gbt_base")?;
    let sizes = take("tree_sizes")?;
    let (feat, thr, left, right, val) = (take("node_feat")?, take("node_thr")?, take("node_left")?, take("node_right")?, take("node_val")?);
    let total: usize = sizes.iter().map(|s| *s as usize).sum();
    if [feat.len(), thr.len(), left.len(), right.len(), val.len()].iter().any(|&n| n != total) {
        return Err("node arrays do not match tree_sizes".into());
    }
    let mut trees = Vec::with_capacity(sizes.len());
    let mut i = 0usize;
    for &size in sizes {
        let size = size as usize;
        let mut t = Vec::with_capacity(size);
        for j in i..i + size {
            let leaf = feat[j] < 0.0;
            t.push(ExportNode {
                feature: if leaf { -1 } else { feat[j] as i32 },
                left: left[j] as u32,
                right: right[j] as u32,
                value: if leaf { val[j] } else { thr[j] },
            });
        }
        trees.push(t);
        i += size;
    }
    gbt::encode(nf, base, &trees).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skato_control::intent::gbt::GbtModel;

    #[test]
    fn a_small_export_converts_and_scores() {
        let text = "LDMODEL 1\nkind cand_gbt\nnf 2\nnb 62\nhidden 0\ngbt_base 1 -0.5\ntree_sizes 2 3 1\n\
                    node_feat 4 1 -1 -1 -1\nnode_thr 4 0.25 0 0 0\nnode_left 4 1 0 0 0\nnode_right 4 2 0 0 0\n\
                    node_val 4 9 2.0 -3.0 0.125\n";
        let bytes = convert_ldm(text).unwrap();
        let m = GbtModel::parse(&bytes).unwrap();
        assert_eq!((m.features(), m.trees(), m.nodes()), (2, 2, 4));
        assert_eq!(m.raw(&[0.0, 0.25]), -0.5 + 2.0 + 0.125);
        assert_eq!(m.raw(&[0.0, 0.5]), -0.5 - 3.0 + 0.125);
        assert!(convert_ldm("LDMODEL 1\nkind mlp\nnf 2\n").is_err());
        assert!(convert_ldm("hello").is_err());
    }
}
