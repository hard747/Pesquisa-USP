#!/usr/bin/env python3
"""
eval_v4_large_gt.py — Evaluate V4 (proposed pipeline) ALONE against the large
ground truth (~562 manual annotations incl. pupil center AND glints).

No ablation, no other versions. Just: run the real EyeTrackingPipeline (v4) on
each annotated session and score pupil-center error + glint error, broken down
by dark/bright frames so we can see exactly where V4 wins or loses.

Usage:
    python eval_v4_large_gt.py
"""

import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

# Reuse helpers + the real pipeline from the ablation module
from ablation_study import (
    _load_gray, _list_session_images, FrameClassifier,
)
from pupil_glint_detector_v4 import EyeTrackingPipeline
from config import REPO_ROOT

GT_LARGE = os.path.join(REPO_ROOT, "gt_builder", "ground_truth_large.json")
OUT_DIR  = os.path.join(REPO_ROOT, "ablation_results")
GLINT_MATCH_THR = 8.0   # px: predicted glint counts as matched to a GT glint


def _match_glints(gt_glints, pred_glints):
    """Greedy nearest-neighbour matching. Returns (matched_dists, n_gt, n_pred)."""
    gt = [tuple(g) for g in gt_glints]
    pred = [tuple(g.center) for g in pred_glints]
    matched = []
    used = set()
    for gx, gy in gt:
        best_d, best_j = None, None
        for j, (px, py) in enumerate(pred):
            if j in used:
                continue
            d = math.hypot(px - gx, py - gy)
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= GLINT_MATCH_THR:
            matched.append(best_d)
            used.add(best_j)
    return matched, len(gt), len(pred)


def process_session(participant, session, gt_map, warmup=40):
    """Run V4 on one session, score against the GT frames in gt_map."""
    session_dir, imgs = _list_session_images(participant, session)
    if not imgs:
        return []

    imgs_set = {fn: i for i, fn in enumerate(imgs)}
    gt_idx = [imgs_set[fn] for fn in gt_map if fn in imgs_set]
    if not gt_idx:
        return []
    start_idx = max(0, min(gt_idx) - warmup)
    end_idx   = min(len(imgs), max(gt_idx) + 5)

    # Frame-classifier prescan
    classifier = FrameClassifier()
    counts = []
    for fn in imgs[:min(200, len(imgs))]:
        g = _load_gray(os.path.join(session_dir, fn))
        if g is not None:
            counts.append(classifier.bright_pixel_count(g))

    pipeline = EyeTrackingPipeline()
    pipeline.classifier.prescan(counts)

    rows = []
    for idx, fname in enumerate(imgs[start_idx:end_idx], start=start_idx):
        gray = _load_gray(os.path.join(session_dir, fname))
        if gray is None:
            continue
        t0 = time.perf_counter()
        result = pipeline.process(gray, fname, idx + 1)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if fname not in gt_map:
            continue
        gt = gt_map[fname]
        is_blink_gt = bool(gt.get("is_blink", False)) or gt.get("pupil_gt") is None

        detected = (not result.is_blink) and (result.pupil is not None)
        if detected and not is_blink_gt:
            pcx, pcy = result.pupil.center
            err = math.hypot(pcx - gt["pupil_gt"][0], pcy - gt["pupil_gt"][1])
        else:
            pcx, pcy = (result.pupil.center if (result.pupil is not None) else (0.0, 0.0))
            err = float("nan")

        # Glint scoring (only on non-blink frames with annotated glints)
        g_matched, g_ngt, g_npred = ([], 0, 0)
        if gt.get("glints") and not gt.get("is_blink"):
            g_matched, g_ngt, g_npred = _match_glints(gt["glints"], result.glints)

        # Capture pupil quality signals whenever a pupil was emitted (even if it
        # ends up being a wrong/blink detection) — needed to tune the reject gate
        p = result.pupil
        rows.append({
            "participant_id": participant,
            "session_id":     session,
            "filename":       fname,
            "frame_type":     gt.get("frame_type", "unknown"),
            "is_blink_gt":    is_blink_gt,
            "gt_cx":          gt["pupil_gt"][0] if not is_blink_gt else float("nan"),
            "gt_cy":          gt["pupil_gt"][1] if not is_blink_gt else float("nan"),
            "pred_cx":        round(pcx, 3),
            "pred_cy":        round(pcy, 3),
            "error_px":       round(err, 3) if not math.isnan(err) else float("nan"),
            "detected":       detected,
            "pred_is_dark":   result.frame_type == "dark",
            "pred_is_blink":  bool(result.is_blink),
            "blink_reason":   result.blink_reason,
            "pupil_emitted":  p is not None,
            "edge_support":   round(p.edge_support, 4) if p is not None else float("nan"),
            "fit_rms":        round(p.fit_rms, 4) if p is not None else float("nan"),
            "n_inliers":      p.n_inliers if p is not None else -1,
            "radius_eq":      round(p.radius_eq, 3) if p is not None else float("nan"),
            "aspect_ratio":   round(p.aspect_ratio, 4) if p is not None else float("nan"),
            "quality":        round(result.quality, 4),
            "glint_n_gt":     g_ngt,
            "glint_n_pred":   g_npred,
            "glint_n_match":  len(g_matched),
            "glint_mae":      round(float(np.mean(g_matched)), 3) if g_matched else float("nan"),
            "time_ms":        round(elapsed, 2),
        })
    return rows


def _worker_init():
    """Each worker uses single-threaded OpenCV to avoid oversubscription."""
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass


def _worker(task):
    (pid, sess), gt_map = task
    try:
        return process_session(pid, sess, gt_map)
    except Exception as e:
        print(f"  ERROR {pid}/{sess}: {e}", flush=True)
        return []


def summarize(df, label, mask=None):
    sub = df if mask is None else df[mask]
    n = len(sub)
    det = sub[sub["detected"]]
    errs = det["error_px"].dropna()
    if len(errs) == 0:
        print(f"  {label:<14} N={n:<4} detected=0  (no scored frames)")
        return
    print(f"  {label:<14} N={n:<4} det={len(det)/max(n,1)*100:5.1f}%  "
          f"MAE={errs.mean():6.2f}  RMSE={np.sqrt((errs**2).mean()):6.2f}  "
          f"P90={np.percentile(errs,90):6.2f}  "
          f"<5px={ (errs<5).mean()*100:5.1f}%  <10px={ (errs<10).mean()*100:5.1f}%")


def main():
    if not os.path.exists(GT_LARGE):
        print("GT not found:", GT_LARGE); sys.exit(1)
    gt = json.load(open(GT_LARGE))
    print(f"Loaded large GT: {len(gt)} annotations")

    # Group: (participant, session) -> {real_filename -> entry}
    sessions = defaultdict(dict)
    for key, v in gt.items():
        pid, sess = v["participant_id"], v["session"]
        prefix = f"{pid}_{sess}_"
        fname = key[len(prefix):] if key.startswith(prefix) else os.path.basename(key)
        sessions[(pid, sess)][fname] = v
    print(f"Sessions with GT: {len(sessions)}")

    tasks = sorted(sessions.items())
    all_rows = []
    n_workers = min(4, len(tasks))
    print(f"Processing {len(tasks)} sessions with {n_workers} parallel workers...",
          flush=True)
    done = 0
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        for rows in pool.imap_unordered(_worker, tasks):
            done += 1
            all_rows.extend(rows)
            print(f"  [{done}/{len(tasks)}] sessions done", flush=True)

    if not all_rows:
        print("No rows collected."); sys.exit(1)

    df = pd.DataFrame(all_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "v4_large_gt_perframe.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 86)
    print(f"  V4 vs LARGE GROUND TRUTH  (N={len(df)} scored frames)")
    print("=" * 86)
    # Pupil metrics exclude blink GT frames (no pupil to score)
    nb = df[~df["is_blink_gt"]]
    print("\n  PUPIL CENTER (excl. blinks):")
    summarize(nb, "ALL")
    summarize(nb, "dark",   nb["frame_type"] == "dark")
    summarize(nb, "bright", nb["frame_type"] == "bright")

    # Blink handling: V4 should NOT emit a pupil on blink GT frames
    bl = df[df["is_blink_gt"]]
    if len(bl):
        correct = int((~bl["detected"]).sum())
        print(f"\n  BLINKS: {correct}/{len(bl)} correctly rejected "
              f"({correct/len(bl)*100:.1f}%)  — {len(bl)-correct} false pupils")

    # Glint metrics
    gdf = df[(df["glint_n_gt"] > 0)]
    tot_gt = int(gdf["glint_n_gt"].sum())
    tot_match = int(gdf["glint_n_match"].sum())
    tot_pred = int(gdf["glint_n_pred"].sum())
    gmae = gdf["glint_mae"].dropna()
    print("\n  GLINTS:")
    print(f"  frames with GT glints: {len(gdf)}   GT glints total: {tot_gt}")
    print(f"  glint recall (matched/GT): {tot_match}/{tot_gt} = "
          f"{tot_match/max(tot_gt,1)*100:.1f}%")
    print(f"  glint precision (matched/pred): {tot_match}/{tot_pred} = "
          f"{tot_match/max(tot_pred,1)*100:.1f}%")
    if len(gmae):
        print(f"  glint MAE (matched, per-frame mean): {gmae.mean():.2f} px")

    # dark/bright glint recall
    for ft in ("dark", "bright"):
        s = gdf[gdf["frame_type"] == ft]
        if len(s):
            m, t = int(s["glint_n_match"].sum()), int(s["glint_n_gt"].sum())
            print(f"    {ft:<6} glint recall: {m}/{t} = {m/max(t,1)*100:.1f}%")

    print(f"\n  mean time: {df['time_ms'].mean():.1f} ms/frame")
    print(f"  per-frame CSV: {out_csv}")
    print("=" * 86)


if __name__ == "__main__":
    main()
