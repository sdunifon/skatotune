"""Train the learned switch veto's candidate-note model (Labs "learned note hold").

Reads the training dumps written by `skato_eval note-intent dump` (vocadito
DEV singers) and `dump-synth` (synthetic lines), builds three candidate rows
per labelled frame (the legal note nearest the decided pitch r and its legal
neighbours, in the chromatic grid and in the file's key) and fits one
gradient-boosted tree classifier for "this candidate is the annotated note".
Writes the model as `LDMODEL 1` text; `skato_eval note-intent convert-model`
turns it into the `.skgb` file the plugin embeds.

Per candidate c the features are the 45 shared ones relative to r (58 with
Studio's lookahead), dc = (r - 100c)/100 and |dc|, 21 pitch summaries
relative to c, |.| of the first 12, and the gaps to c's legal neighbours
(plus 9 lookahead summaries relative to c in Studio): 82 features Live, 104
Studio, in the order `skato_control::intent::features::candidate` builds them.

Labels: frames inside an annotator-1 note where both annotators round to the
same legal note. TEST singers are never read (the dump only holds the split
it was asked for); never train on a TEST dump.

usage:
  python3 train_veto.py --dev live_hq_dev.f32 --synth live_hq_synth.f32 --out live_hq.ldm [--folds]

The shipped models (`crates/skato_control/models`) are this script with the
defaults on `dump --split dev` and `dump-synth --n 400 --seed0 50000`, with
numpy 2.3.1 and scikit-learn 1.7.1 (another scikit-learn may give a slightly
different ensemble: re-run `skato_eval note-intent` on DEV before shipping it).
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Shared features that get a candidate-relative version, in order.
REL = ["m3", "m6", "m12", "m25", "m50", "m100", "m200", "md12", "md25", "md50", "md100", "md200",
       "ps", "pc", "ctr", "o50_100", "o100_200", "mn25", "mx25", "mn100", "mx100"]
REL_F = ["f3", "f6", "f12", "f25", "fall", "fmd", "fmn", "fmx", "flast"]
NSH = 45
NFC_LIVE = NSH + 2 + len(REL) + 12 + 2
NFC_STUDIO = 58 + 2 + len(REL) + 12 + 2 + len(REL_F)


def load(path):
    """A dump: header columns (dict) and the feature block."""
    meta = json.load(open(os.path.splitext(path)[0] + ".json"))
    if meta["split"] == "test":
        raise SystemExit("refusing to train on the TEST split")
    a = np.fromfile(path, dtype="<f4").reshape(meta["rows"], meta["ncol"])
    cols = meta["columns"]
    nhead = cols.index("m3")
    d = {c: a[:, i] for i, c in enumerate(cols[:nhead])}
    return d, a[:, nhead:], cols[nhead:], meta


def legal(n, mask):
    return ((mask.astype(np.int64) >> (n.astype(np.int64) % 12)) & 1) == 1


def candidates(r, mask):
    """[N,3] candidate notes (below, nearest, above), as the corrector's
    TargetSet: the nearest legal note (ties to the lower), its legal neighbours."""
    rel = r / 100.0
    n0 = np.floor(rel + 0.5).astype(np.int64)
    near = np.full(len(r), -1, dtype=np.int64)
    for k in range(0, 12):
        lo, hi = n0 - k, n0 + k
        lo_ok, hi_ok = legal(lo, mask), legal(hi, mask)
        both = lo_ok & hi_ok & (near < 0)
        dl = np.abs(r - 100 * lo)
        dh = np.abs(r - 100 * hi)
        near = np.where(both, np.where(dl <= dh, lo, hi), near)
        near = np.where((near < 0) & lo_ok & ~hi_ok, lo, near)
        near = np.where((near < 0) & hi_ok & ~lo_ok, hi, near)
    below = np.full(len(r), -1, dtype=np.int64)
    above = np.full(len(r), -1, dtype=np.int64)
    for k in range(12, 0, -1):
        below = np.where(legal(near - k, mask), near - k, below)
        above = np.where(legal(near + k, mask), near + k, above)
    return np.stack([below, near, above], 1)


def gaps(c, mask):
    """Semitones to c's legal neighbour below / above (the other side when
    one is missing, then 1)."""
    dn = np.zeros(len(c))
    up = np.zeros(len(c))
    for k in range(12, 0, -1):
        dn = np.where(legal(c - k, mask), k, dn)
        up = np.where(legal(c + k, mask), k, up)
    dn2 = np.where(dn > 0, dn, np.where(up > 0, up, 1.0))
    up2 = np.where(up > 0, up, np.where(dn > 0, dn, 1.0))
    return dn2, up2


def cand_features(F, names, r, c, mask, studio):
    dc = (r - 100.0 * c) / 100.0
    rel = np.stack([F[:, names.index(k)] + dc for k in REL], 1)
    gl, gh = gaps(c, mask)
    nsh = 58 if studio else NSH
    parts = [F[:, :nsh], dc[:, None], np.abs(dc)[:, None], rel, np.abs(rel[:, :12]), gl[:, None], gh[:, None]]
    if studio:
        parts.append(np.stack([F[:, names.index(k)] + dc for k in REL_F], 1))
    return np.concatenate(parts, 1).astype(np.float32)


def build(path, sub, legal_sets, studio):
    """Candidate rows of a dump: X, y (0/1), singer per row."""
    d, F, names, _ = load(path)
    valid = d["valid"] > 0
    out = []
    for leg in legal_sets:
        if leg == "chrom":
            mask = np.full(len(d["r"]), 0xFFF)
            g, g2 = d["gt_chrom"], d["gt2_chrom"]
        else:
            mask = np.nan_to_num(d["key_mask"]).astype(np.int64)
            g, g2 = d["gt_key"], d["gt2_key"]
        lab = valid & np.isfinite(g) & (g == g2) & (mask > 0)
        rows = np.where(lab)[0][::sub]
        if len(rows) == 0:
            continue
        r = d["r"][rows].astype(np.float64)
        C = candidates(r, mask[rows])
        for j in range(3):
            c = C[:, j]
            ok = c >= 0
            X = cand_features(F[rows], names, r, np.where(ok, c, 0), mask[rows], studio)
            y = (c == g[rows]).astype(np.int8)
            out.append((X[ok], y[ok], d["singer"][rows][ok].astype(int)))
    return (np.concatenate([o[i] for o in out]) for i in range(3))


def fit(X, y, args):
    m = HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=args.lr, max_leaf_nodes=args.leaves,
                                       min_samples_leaf=200, l2_regularization=1.0, random_state=args.seed)
    m.fit(X, y)
    return m


def export(path, m, nf):
    """`LDMODEL 1` text (`skato_eval note-intent convert-model` reads it)."""
    lines = ["LDMODEL 1", "kind cand_gbt", f"nf {nf}", "nb 62", "hidden 0"]
    base = float(np.ravel(m._baseline_prediction)[0])
    feats, thr, left, right, val, sizes = [], [], [], [], [], []
    for pp in m._predictors:
        nodes = pp[0].nodes
        sizes.append(len(nodes))
        for nd in nodes:
            leaf = bool(nd["is_leaf"])
            feats.append(-1 if leaf else int(nd["feature_idx"]))
            thr.append(float(nd["num_threshold"]))
            left.append(int(nd["left"]))
            right.append(int(nd["right"]))
            val.append(float(nd["value"]))

    def arr(name, a, fmt):
        lines.append(f"{name} {len(a)} " + " ".join(fmt(x) for x in a))
    arr("gbt_base", [base], lambda x: f"{x:.9g}")
    arr("tree_sizes", sizes, str)
    arr("node_feat", feats, str)
    arr("node_thr", thr, lambda x: f"{x:.9g}")
    arr("node_left", left, str)
    arr("node_right", right, str)
    arr("node_val", val, lambda x: f"{x:.9g}")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dev", required=True, help="dump of the DEV split (skato_eval note-intent dump)")
    ap.add_argument("--synth", help="synthetic dump (skato_eval note-intent dump-synth)")
    ap.add_argument("--out", required=True, help="output LDMODEL text")
    ap.add_argument("--folds", action="store_true",
                    help="also write one model per held-out DEV singer (OUT_fold<singer>.ldm) for honest DEV numbers")
    ap.add_argument("--sub", type=int, default=2, help="every n-th labelled DEV frame")
    ap.add_argument("--synth_sub", type=int, default=4, help="every n-th labelled synthetic frame")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    meta = json.load(open(os.path.splitext(args.dev)[0] + ".json"))
    studio = meta["config"].startswith("studio")
    nf = NFC_STUDIO if studio else NFC_LIVE
    X, y, singer = build(args.dev, args.sub, ("chrom", "key"), studio)
    if args.synth:
        Xs, ys, _ = build(args.synth, args.synth_sub, ("chrom",), studio)
        print(f"+ {len(ys)} synthetic candidate rows")
    print(f"{meta['config']}: {len(y)} DEV candidate rows, {nf} features [{time.time() - t0:.0f} s]", flush=True)

    def train(rows):
        Xt, yt = X[rows], y[rows]
        if args.synth:
            Xt, yt = np.concatenate([Xt, Xs]), np.concatenate([yt, ys])
        return fit(Xt, yt, args)

    if args.folds:
        stem = os.path.splitext(args.out)[0]
        for s in sorted(set(singer.tolist())):
            path = f"{stem}_fold{meta['singers'][s]}.ldm"
            export(path, train(singer != s), nf)
            print(f"  {path} [{time.time() - t0:.0f} s]", flush=True)
    export(args.out, train(np.ones(len(y), dtype=bool)), nf)
    print(f"wrote {args.out} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
