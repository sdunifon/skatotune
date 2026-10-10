//! Gradient-boosted tree ensembles for the learned switch veto: a compact,
//! versioned and checksummed binary format, embedded with `include_bytes!`
//! and parsed once (outside the audio thread) into a flat node array.
//!
//! # Format `SKGB` version 1 (little-endian)
//!
//! | bytes | field |
//! |---|---|
//! | 4 | magic `SKGB` |
//! | 4 | version (`u32`, 1) |
//! | 4 | features per candidate (`u32`) |
//! | 4 | trees (`u32`) |
//! | 4 | nodes in total (`u32`, at most 65 535) |
//! | 4 | base score (`f32`) |
//! | 4 per tree | nodes in each tree (`u32`) |
//! | 10 per node | feature (`i16`, −1 = leaf), left and right child (`u16`, relative to the tree's first node), value (`f32`: the split threshold, or the leaf's score) |
//! | 8 | FNV-1a 64 of every byte before it |
//!
//! A split sends `x[feature] <= threshold` to the left child (the
//! scikit-learn `HistGradientBoostingClassifier` rule for values that are
//! present; the features are always finite). The ensemble's raw score is the
//! base score plus one leaf per tree, summed in tree order in `f32`, so the
//! result is bit-identical to the prototype that the evaluation used.
//!
//! Every internal node's children lie inside its own tree and after it, so a
//! walk ends at a leaf within the tree's node count: [`GbtModel::raw`] is
//! bounded and never indexes out of range.

/// Magic bytes and version of the format.
const MAGIC: &[u8; 4] = b"SKGB";
const VERSION: u32 = 1;
/// Feature index of a leaf in memory.
const LEAF: u16 = u16::MAX;
/// Header bytes before the per-tree sizes.
const HEADER: usize = 24;
/// Bytes per node on disk.
const NODE_BYTES: usize = 10;

/// Why a model could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// Not an `SKGB` file, or a version this code cannot read.
    Format,
    /// The checksum does not match (truncated or corrupted).
    Checksum,
    /// Sizes, indices or values that do not describe a valid ensemble.
    Invalid,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            ModelError::Format => "not an SKGB version 1 model",
            ModelError::Checksum => "model checksum mismatch",
            ModelError::Invalid => "invalid tree ensemble",
        })
    }
}

impl std::error::Error for ModelError {}

/// One node in memory (12 bytes). Children are absolute indices into
/// [`GbtModel::nodes`].
#[derive(Clone, Copy, Debug, PartialEq)]
struct Node {
    /// Split threshold, or the leaf's score.
    value: f32,
    /// Feature index, or [`LEAF`].
    feature: u16,
    left: u16,
    right: u16,
}

/// A parsed tree ensemble (see the module docs).
#[derive(Clone, Debug)]
pub struct GbtModel {
    nf: usize,
    base: f32,
    /// First node of each tree.
    roots: Vec<u16>,
    nodes: Vec<Node>,
    /// Longest walk any tree can take (its node count).
    steps: usize,
    checksum: u64,
}

/// FNV-1a 64.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], ModelError> {
        let s = self.b.get(self.i..self.i + N).ok_or(ModelError::Invalid)?;
        self.i += N;
        s.try_into().map_err(|_| ModelError::Invalid)
    }
    fn u32(&mut self) -> Result<u32, ModelError> {
        self.take::<4>().map(u32::from_le_bytes)
    }
    fn u16(&mut self) -> Result<u16, ModelError> {
        self.take::<2>().map(u16::from_le_bytes)
    }
    fn i16(&mut self) -> Result<i16, ModelError> {
        self.take::<2>().map(i16::from_le_bytes)
    }
    fn f32(&mut self) -> Result<f32, ModelError> {
        self.take::<4>().map(f32::from_le_bytes)
    }
}

impl GbtModel {
    /// Parse and validate an `SKGB` model (allocates; never call it on the
    /// audio thread).
    pub fn parse(bytes: &[u8]) -> Result<GbtModel, ModelError> {
        if bytes.len() < HEADER + 8 || bytes.get(..4) != Some(MAGIC.as_slice()) {
            return Err(ModelError::Format);
        }
        let (body, sum) = bytes.split_at(bytes.len() - 8);
        let stored = u64::from_le_bytes(sum.try_into().map_err(|_| ModelError::Format)?);
        let mut r = Reader { b: body, i: 4 };
        if r.u32()? != VERSION {
            return Err(ModelError::Format);
        }
        if fnv1a64(body) != stored {
            return Err(ModelError::Checksum);
        }
        let nf = r.u32()? as usize;
        let n_trees = r.u32()? as usize;
        let n_nodes = r.u32()? as usize;
        let base = r.f32()?;
        let expected = HEADER + 4 * n_trees + NODE_BYTES * n_nodes;
        if nf == 0 || nf >= usize::from(LEAF) || n_trees == 0 || n_nodes > usize::from(u16::MAX) || body.len() != expected {
            return Err(ModelError::Invalid);
        }
        if !base.is_finite() {
            return Err(ModelError::Invalid);
        }
        let mut sizes = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            sizes.push(r.u32()? as usize);
        }
        if sizes.contains(&0) || sizes.iter().sum::<usize>() != n_nodes {
            return Err(ModelError::Invalid);
        }
        let mut roots = Vec::with_capacity(n_trees);
        let mut nodes = Vec::with_capacity(n_nodes);
        let mut start = 0usize;
        for &size in &sizes {
            roots.push(u16::try_from(start).map_err(|_| ModelError::Invalid)?);
            for j in 0..size {
                let feature = r.i16()?;
                let (left, right) = (usize::from(r.u16()?), usize::from(r.u16()?));
                let value = r.f32()?;
                if !value.is_finite() {
                    return Err(ModelError::Invalid);
                }
                let node = if feature < 0 {
                    Node { value, feature: LEAF, left: 0, right: 0 }
                } else {
                    // Children inside this tree and after this node, so every
                    // walk terminates at a leaf.
                    let f = feature as usize;
                    if f >= nf || left <= j || right <= j || left >= size || right >= size {
                        return Err(ModelError::Invalid);
                    }
                    let abs = |c: usize| u16::try_from(start + c).map_err(|_| ModelError::Invalid);
                    Node { value, feature: feature as u16, left: abs(left)?, right: abs(right)? }
                };
                nodes.push(node);
            }
            start += size;
        }
        let steps = sizes.iter().copied().max().unwrap_or(0);
        Ok(GbtModel { nf, base, roots, nodes, steps, checksum: stored })
    }

    /// Features per candidate the model was trained on.
    pub fn features(&self) -> usize {
        self.nf
    }

    pub fn trees(&self) -> usize {
        self.roots.len()
    }

    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The file's FNV-1a 64 checksum (identifies the model in its README).
    pub fn checksum(&self) -> u64 {
        self.checksum
    }

    /// Raw ensemble score (log-odds) of a feature vector. Allocation-free and
    /// bounded: one walk of at most [`Self::steps`] nodes per tree. A missing
    /// feature reads as 0.
    pub fn raw(&self, x: &[f32]) -> f32 {
        let mut raw = self.base;
        for &root in &self.roots {
            let mut i = usize::from(root);
            for _ in 0..self.steps {
                let Some(n) = self.nodes.get(i) else { break };
                if n.feature == LEAF {
                    raw += n.value;
                    break;
                }
                let v = x.get(usize::from(n.feature)).copied().unwrap_or(0.0);
                i = usize::from(if v <= n.value { n.left } else { n.right });
            }
        }
        raw
    }
}

/// One tree node as the training export describes it (offline use: model
/// conversion and tests).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExportNode {
    /// Feature index, or a negative number for a leaf.
    pub feature: i32,
    /// Children, relative to the tree's first node (ignored for a leaf).
    pub left: u32,
    pub right: u32,
    /// Split threshold, or the leaf's score.
    pub value: f32,
}

/// Write an ensemble in the `SKGB` format (offline use: `skato_eval
/// note-intent convert-model`). `trees` holds each tree's nodes in the order
/// the export lists them. The result is checked by [`GbtModel::parse`].
pub fn encode(nf: usize, base: f32, trees: &[Vec<ExportNode>]) -> Result<Vec<u8>, ModelError> {
    let n_nodes: usize = trees.iter().map(Vec::len).sum();
    let to_u32 = |v: usize| u32::try_from(v).map_err(|_| ModelError::Invalid);
    let mut w = Vec::with_capacity(HEADER + 4 * trees.len() + NODE_BYTES * n_nodes + 8);
    w.extend_from_slice(MAGIC);
    w.extend_from_slice(&VERSION.to_le_bytes());
    w.extend_from_slice(&to_u32(nf)?.to_le_bytes());
    w.extend_from_slice(&to_u32(trees.len())?.to_le_bytes());
    w.extend_from_slice(&to_u32(n_nodes)?.to_le_bytes());
    w.extend_from_slice(&base.to_le_bytes());
    for t in trees {
        w.extend_from_slice(&to_u32(t.len())?.to_le_bytes());
    }
    for t in trees {
        for n in t {
            let feature = if n.feature < 0 { -1i16 } else { i16::try_from(n.feature).map_err(|_| ModelError::Invalid)? };
            let child = |c: u32| if n.feature < 0 { Ok(0u16) } else { u16::try_from(c).map_err(|_| ModelError::Invalid) };
            w.extend_from_slice(&feature.to_le_bytes());
            w.extend_from_slice(&child(n.left)?.to_le_bytes());
            w.extend_from_slice(&child(n.right)?.to_le_bytes());
            w.extend_from_slice(&n.value.to_le_bytes());
        }
    }
    let sum = fnv1a64(&w);
    w.extend_from_slice(&sum.to_le_bytes());
    GbtModel::parse(&w)?;
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(value: f32) -> ExportNode {
        ExportNode { feature: -1, left: 0, right: 0, value }
    }

    fn split(feature: i32, value: f32, left: u32, right: u32) -> ExportNode {
        ExportNode { feature, left, right, value }
    }

    /// Two stumps and a depth-2 tree.
    fn small() -> Vec<u8> {
        let trees = vec![
            vec![split(0, 0.5, 1, 2), leaf(-1.0), leaf(1.0)],
            vec![split(1, -2.0, 1, 2), leaf(0.25), leaf(-0.25)],
            vec![split(0, 0.0, 1, 2), split(1, 0.0, 3, 4), leaf(10.0), leaf(100.0), leaf(1000.0)],
        ];
        encode(2, 0.5, &trees).unwrap()
    }

    #[test]
    fn scores_follow_the_split_rule() {
        let m = GbtModel::parse(&small()).unwrap();
        assert_eq!((m.features(), m.trees(), m.nodes()), (2, 3, 11));
        // x0 = 0.5 goes left (<=) at the first stump, right at the third.
        assert_eq!(m.raw(&[0.5, 0.0]), 0.5 - 1.0 - 0.25 + 10.0);
        assert_eq!(m.raw(&[-1.0, -3.0]), 0.5 - 1.0 + 0.25 + 100.0);
        assert_eq!(m.raw(&[-1.0, 1.0]), 0.5 - 1.0 - 0.25 + 1000.0);
        // NaN goes right; a missing feature reads as 0.
        assert_eq!(m.raw(&[f32::NAN, f32::NAN]), 0.5 + 1.0 - 0.25 + 10.0);
        assert_eq!(m.raw(&[]), 0.5 - 1.0 - 0.25 + 100.0);
    }

    #[test]
    fn corrupt_models_are_rejected() {
        let good = small();
        assert!(GbtModel::parse(&good).is_ok());
        assert_eq!(GbtModel::parse(&good[..good.len() - 1]).unwrap_err(), ModelError::Checksum);
        assert_eq!(GbtModel::parse(b"LDMODEL 1").unwrap_err(), ModelError::Format);
        assert_eq!(GbtModel::parse(&[]).unwrap_err(), ModelError::Format);
        for i in 4..good.len() - 8 {
            let mut bad = good.clone();
            bad[i] ^= 0x10;
            assert!(GbtModel::parse(&bad).is_err(), "flipped byte {i}");
        }
        // A child that points back up (a loop) is invalid even with a good checksum.
        let looped = vec![vec![split(0, 0.5, 0, 2), leaf(1.0), leaf(2.0)]];
        assert_eq!(encode(1, 0.0, &looped).unwrap_err(), ModelError::Invalid);
        let outside = vec![vec![split(0, 0.5, 1, 3), leaf(1.0), leaf(2.0)]];
        assert_eq!(encode(1, 0.0, &outside).unwrap_err(), ModelError::Invalid);
        let feature = vec![vec![split(5, 0.5, 1, 2), leaf(1.0), leaf(2.0)]];
        assert_eq!(encode(2, 0.0, &feature).unwrap_err(), ModelError::Invalid);
        let nan = vec![vec![leaf(f32::NAN)]];
        assert_eq!(encode(2, 0.0, &nan).unwrap_err(), ModelError::Invalid);
    }
}
