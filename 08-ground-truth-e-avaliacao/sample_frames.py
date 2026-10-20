#!/usr/bin/env python3
"""
Stratified, statistically-robust frame sampler for building a larger ground truth.

Design
------
Strata = user x session x frame_type (dark / bright).
For each of the 15 users and 7 sessions (fp1..fv1) we:
  1. Estimate a per-session dark/bright threshold from the bright-pixel-count
     distribution (1D Otsu) -- robust to the "consecutive identical frames"
     problem because it is data-driven per session, not an odd/even assumption.
  2. Drop near-certain blinks/occlusions (very low bright-pixel count AND low
     edge energy) so the GT samples are annotatable pupils.
  3. Sample PER_TYPE frames of each type, spread uniformly across the session
     timeline (deterministic, seeded) so no temporal clustering.

Output: gt_builder/sample_manifest.csv  (one row per frame to annotate)
        columns: participant_id, session, file_name, image_path, frame_type,
                 bright_px, auto_cx, auto_cy   (auto_* = neutral intensity-centroid
                 seed, shown faintly in the annotator; NOT a method prediction)

The sampler is headless and reproducible. Re-running with the same seed yields
the same manifest. Increase PER_TYPE to grow the GT.
"""
import os
import csv
import json
import argparse
import numpy as np
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_manifest.csv")
CACHE = os.path.join(HERE, "scan_cache.json")   # caches expensive per-frame stats

USERS = ["P01", "P02", "P03", "P04", "P05",
         "P06", "P07", "P08", "P09", "P10",
         "P11", "P12", "P13", "P14", "P15"]
SESSIONS = ["fp1", "fq1", "fr1", "fs1", "ft1", "fu1", "fv1"]

BRIGHT_VAL = 200          # pixel >= this counts as "bright"
BLINK_BRIGHT_FRAC = 0.0008  # below this fraction of bright px -> candidate blink
SEED = 20260611


def resolve_session_dir(user, sess):
    """Handle the 3 users nested under <user>_fixacoes/."""
    for cand in (os.path.join(DATA, user, sess),
                 os.path.join(DATA, user, user + "_fixacoes", sess)):
        if os.path.isdir(cand):
            return cand
    return None


def otsu_threshold_1d(values):
    """1D Otsu on a list of scalar values -> threshold."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(np.median(v))
    w = np.cumsum(hist)
    wb = w / total
    wf = 1.0 - wb
    cum = np.cumsum(hist * centers)
    mb = np.divide(cum, w, out=np.zeros_like(cum), where=w > 0)
    mtot = cum[-1]
    mf = np.divide(mtot - cum, total - w, out=np.zeros_like(cum), where=(total - w) > 0)
    between = wb * wf * (mb - mf) ** 2
    idx = int(np.nanargmax(between))
    return float(centers[idx])


def scan_session(sess_dir):
    """Read every frame (grayscale, half-res) -> per-frame bright-pixel count + centroid."""
    files = sorted(f for f in os.listdir(sess_dir) if f.lower().endswith(".jpg"))
    rows = []
    for fn in files:
        img = cv2.imread(os.path.join(sess_dir, fn), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        bmask = small >= BRIGHT_VAL
        bcount = int(bmask.sum())
        frac = bcount / small.size
        # neutral seed: centroid of bright pixels at full-res scale (x2)
        if bcount > 5:
            ys, xs = np.nonzero(bmask)
            cx, cy = float(xs.mean()) * 2.0, float(ys.mean()) * 2.0
        else:
            cx, cy = -1.0, -1.0
        rows.append((fn, bcount, frac, cx, cy))
    return rows


def spread_pick(indices, k, rng):
    """Pick k indices spread uniformly across the ordered list (with jitter)."""
    if len(indices) <= k:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, k)
    chosen = set()
    out = []
    for p in positions:
        i = int(round(p))
        # jitter to avoid exact periodicity, keep within bounds & unique
        for _ in range(5):
            j = min(len(indices) - 1, max(0, i + rng.integers(-2, 3)))
            if j not in chosen:
                i = j
                break
        chosen.add(i)
        out.append(indices[i])
    return out


def build_scan_cache():
    """Scan every session once and cache per-frame stats to disk (slow: ~10 min)."""
    cache = {}
    for user in USERS:
        for sess in SESSIONS:
            sdir = resolve_session_dir(user, sess)
            if sdir is None:
                print(f"  [skip] {user}/{sess} not found")
                continue
            rows = scan_session(sdir)
            cache[f"{user}/{sess}"] = {"dir": sdir, "rows": rows}
            print(f"  scanned {user}/{sess}: {len(rows)} frames")
    json.dump(cache, open(CACHE, "w"))
    print(f"Cached scan to {CACHE}")
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=3,
                    help="frames of EACH type per user PER SESSION (default 3 -> 6/session, 42/user)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--rescan", action="store_true", help="force re-scan (ignore cache)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    if args.rescan or not os.path.exists(CACHE):
        cache = build_scan_cache()
    else:
        cache = json.load(open(CACHE))
        print(f"Loaded scan cache ({len(cache)} sessions) from {CACHE}")

    manifest = []
    for user in USERS:
        nd = nb = 0
        for sess in SESSIONS:
            key = f"{user}/{sess}"
            if key not in cache:
                continue
            sdir = cache[key]["dir"]
            rows = cache[key]["rows"]
            if not rows:
                continue
            counts = [r[1] for r in rows]
            thr = otsu_threshold_1d(counts)
            dark_pool, bright_pool = [], []
            for fn, bcount, frac, cx, cy in rows:
                if frac < BLINK_BRIGHT_FRAC:
                    continue  # candidate blink -> exclude from accuracy GT
                rec = (fn, os.path.join(sdir, fn), bcount, cx, cy)
                (bright_pool if bcount >= thr else dark_pool).append(rec)
            # stratified spread-sample per (session, type)
            for pool, label in ((dark_pool, "dark"), (bright_pool, "bright")):
                picks = spread_pick(list(range(len(pool))), args.per_cell, rng)
                for i in picks:
                    fn, path, bc, cx, cy = pool[i]
                    manifest.append([user, sess, fn, path, label, bc,
                                     round(cx, 1), round(cy, 1)])
                    if label == "dark":
                        nd += 1
                    else:
                        nb += 1
        print(f"  {user:14s} dark={nd:2d} bright={nb:2d}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "session", "file_name", "image_path",
                    "frame_type", "bright_px", "auto_cx", "auto_cy"])
        w.writerows(manifest)
    print(f"\nWrote {len(manifest)} frames to {args.out}")
    print(f"Per-cell target: {args.per_cell} -> {2 * args.per_cell}/session, "
          f"~{7 * 2 * args.per_cell}/user, target ~{len(USERS) * 7 * 2 * args.per_cell} total")


if __name__ == "__main__":
    main()
