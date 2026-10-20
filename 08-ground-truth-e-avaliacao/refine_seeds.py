#!/usr/bin/env python3
"""
Refine the neutral crop-seed for each manifest frame using temporal differencing.

The bright-pixel centroid is unreliable (ambient-lit skin dominates). The pupil,
however, is exactly what changes between a dark frame and its bright neighbour, so
|frame - neighbour| isolates the pupil region. We take the centroid of the largest
high-difference blob as the seed. This only re-centres the annotation crop; it is
NOT used as ground truth (the annotator still clicks the true center).

Updates sample_manifest.csv in place (auto_cx, auto_cy).
"""
import os
import csv
import re
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "sample_manifest.csv")


def neighbour_path(image_path):
    """Return path of the adjacent frame (number +/-1) in the same folder."""
    d = os.path.dirname(image_path)
    fn = os.path.basename(image_path)
    m = re.match(r"(.*?)(\d+)(\.\w+)$", fn)
    if not m:
        return None
    prefix, num, ext = m.group(1), m.group(2), m.group(3)
    width = len(num)
    n = int(num)
    for cand in (n + 1, n - 1):
        if cand < 0:
            continue
        p = os.path.join(d, f"{prefix}{cand:0{width}d}{ext}")
        if os.path.exists(p):
            return p
    return None


def refine_seed(image_path):
    cur = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if cur is None:
        return None
    nb = neighbour_path(image_path)
    if nb is None:
        return None
    other = cv2.imread(nb, cv2.IMREAD_GRAYSCALE)
    if other is None or other.shape != cur.shape:
        return None
    diff = cv2.absdiff(cur, other)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)
    thr = max(25, int(np.percentile(diff, 99)))
    mask = (diff >= thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    # largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    cx, cy = cents[k]
    return round(float(cx), 1), round(float(cy), 1)


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    fields = rows[0].keys()
    refined = 0
    for r in rows:
        s = refine_seed(r["image_path"])
        if s:
            r["auto_cx"], r["auto_cy"] = s
            refined += 1
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    print(f"Refined {refined}/{len(rows)} seeds via temporal differencing -> {MANIFEST}")


if __name__ == "__main__":
    main()
