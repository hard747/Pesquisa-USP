#!/usr/bin/env python3
"""
pupil_glint_detector_v4.py
Sub-Pixel Pupil & Glint Detection — Enhanced Stroboscopic P-CR Pipeline v4


Improvements over v2
--------------------
K1  PupilKalmanTracker  (5-state constant-velocity Kalman filter)
    - State: [x, y, r, vx, vy]
    - Annular-band search seeded from Kalman-predicted centre,
      not from the (potentially stale/drifted) temporal-diff ROI centroid.
    - Mahalanobis innovation gate (chi², 3 DOF, p=0.999) rejects
      catastrophic single-frame failures automatically.
    - Saccade vs. outlier discrimination:
        large Mahalanobis + good quality  → saccade  → reinitialise tracker
        large Mahalanobis + poor quality  → outlier  → treat as blink

K2  PROSAC-inspired sampling
    - Edge points sorted by gradient magnitude before RANSAC.
    - Each iteration draws 3 "elite" (top-25 %) + 2 "standard" points,
      biasing toward high-gradient (more reliable) boundary locations.

K3  Tighter radial gradient filter  (cos θ > 0.50, was 0.35 ≈ cos 69°)
    - Reduces false-positive iris edges in the annular band.

K4  Doubled RANSAC iterations (100, was 50).

K5  Prescan always called before offline batch processing.

K6  Synthetic ROI from Kalman prediction when temporal-diff ROI is absent.

K7  Inlier distance tightened to 1.5 px (was 2.0 px) after RANSAC convergence
    is more reliable with PROSAC.

All output columns identical to v2; CSV is drop-in compatible with
pipeline_15_participantes.py.
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NATIVE_W, NATIVE_H = 640, 480
BRIGHT_FLOOR       = 200

# Mahalanobis gates (chi², 3 DOF)
_CHI2_95  = 7.815
_CHI2_999 = 16.27   # used as main gate — generous to not flag saccades as outliers

# Maximum pupil displacement considered a saccade (not outlier)
_SACCADE_DIST_PX = 80.0


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlintResult:
    center: Tuple[float, float]
    area: float
    peak_intensity: float
    circularity: float


@dataclass
class PupilResult:
    center: Tuple[float, float]
    ellipse: Tuple
    radius_eq: float
    aspect_ratio: float
    edge_support: float
    fit_rms: float
    n_inliers: int


@dataclass
class FrameResult:
    filename: str
    frame_idx: int
    frame_type: str
    is_blink: bool
    blink_reason: str = ""
    pupil: Optional[PupilResult] = None
    glints: List[GlintResult] = field(default_factory=list)
    glint_ref: Optional[Tuple[float, float]] = None
    vector: Optional[Tuple[float, float]] = None
    quality: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# S1 — ADAPTIVE FRAME CLASSIFIER  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

class FrameClassifier:
    """Largest-gap threshold on rolling bright-pixel counts."""

    MIN_HISTORY  = 10
    HISTORY_SIZE = 90

    def __init__(self) -> None:
        self._counts: deque = deque(maxlen=self.HISTORY_SIZE)
        self._adaptive_thresh: Optional[float] = None

    def prescan(self, all_counts: list) -> None:
        self._adaptive_thresh = self._largest_gap_threshold(
            np.array(all_counts, dtype=np.float32))

    def classify(self, gray: np.ndarray) -> str:
        count = int(np.sum(gray >= BRIGHT_FLOOR))
        self._counts.append(count)
        if len(self._counts) >= self.MIN_HISTORY:
            self._update_threshold()
        if self._adaptive_thresh is None:
            if len(self._counts) >= 2:
                mu = np.mean(list(self._counts)[:-1])
                return "bright" if count > mu * 1.3 else "dark"
            return "unknown"
        return "bright" if count >= self._adaptive_thresh else "dark"

    @staticmethod
    def _largest_gap_threshold(data: np.ndarray) -> Optional[float]:
        if data.max() - data.min() < 10:
            return None
        s   = np.sort(data)
        idx = int(np.argmax(np.diff(s)))
        return float((s[idx] + s[idx + 1]) / 2.0)

    def _update_threshold(self) -> None:
        t = self._largest_gap_threshold(np.array(self._counts, dtype=np.float32))
        if t is not None:
            self._adaptive_thresh = t

    def bright_pixel_count(self, gray: np.ndarray) -> int:
        return int(np.sum(gray >= BRIGHT_FLOOR))


# ─────────────────────────────────────────────────────────────────────────────
# S2 — ROI ESTIMATOR  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

class ROIEstimator:
    """Temporal-difference ROI with history fallback."""

    HISTORY_SIZE = 5
    MIN_ROI_AREA = 150

    _K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    _K_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,  (7,  7))

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=self.HISTORY_SIZE)
        self._last_valid: Optional[np.ndarray] = None
        self._last_r0_hint: float = 25.0

    def estimate_from_pair(self,
                           dark_gray: np.ndarray,
                           bright_gray: np.ndarray) -> Optional[np.ndarray]:
        diff    = cv2.absdiff(bright_gray, dark_gray)
        blurred = cv2.GaussianBlur(diff, (11, 11), 0)

        thr_pct = float(np.percentile(blurred, 99.0))
        floor   = min(70.0, float(blurred.max()) * 0.70)
        thr     = max(thr_pct, floor)
        _, mask = cv2.threshold(blurred, thr, 255, cv2.THRESH_BINARY)

        raw_area = float(np.sum(mask > 0))
        self._last_r0_hint = (float(math.sqrt(raw_area / math.pi))
                              if raw_area > 0 else 25.0)

        mask = self._morphological_cleanup(mask)
        mask = self._keep_largest_cc(mask)

        if int(np.sum(mask > 0)) < self.MIN_ROI_AREA:
            return self.get_fallback()

        self._last_valid = mask.copy()
        self._history.append(mask.copy())
        return mask

    def get_fallback(self) -> Optional[np.ndarray]:
        if not self._history:
            return self._last_valid
        combined = np.zeros_like(self._history[0])
        for m in self._history:
            combined = cv2.bitwise_or(combined, m)
        return combined

    def dilate_for_pupil(self, roi: np.ndarray, px: int = 40) -> np.ndarray:
        return self._bbox_expand(roi, px)

    def dilate_for_glints(self, roi: np.ndarray, px: int = 65) -> np.ndarray:
        return self._bbox_expand(roi, px)

    @staticmethod
    def _bbox_expand(roi: np.ndarray, px: int) -> np.ndarray:
        h, w = roi.shape
        ys, xs = np.where(roi > 0)
        if len(xs) == 0:
            cx, cy = w // 2, h // 2
            x1, x2 = max(0, cx - px), min(w, cx + px)
            y1, y2 = max(0, cy - px), min(h, cy + px)
        else:
            x1 = max(0,  int(xs.min()) - px)
            x2 = min(w,  int(xs.max()) + px)
            y1 = max(0,  int(ys.min()) - px)
            y2 = min(h,  int(ys.max()) + px)
        out = np.zeros_like(roi)
        out[y1:y2, x1:x2] = 255
        return out

    @staticmethod
    def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                ROIEstimator._K_CLOSE, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                ROIEstimator._K_OPEN,  iterations=1)
        return mask

    @staticmethod
    def _keep_largest_cc(mask: np.ndarray) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(mask)
        out[labels == largest] = 255
        return out


# ─────────────────────────────────────────────────────────────────────────────
# S3 + S4 — GLINT DETECTOR  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

class GlintDetector:
    """Intensity-weighted sub-pixel centroid glint detection."""

    MIN_AREA          = 2
    MAX_AREA          = 400
    MIN_CIRCULARITY   = 0.45
    MIN_PEAK_DARK     = 200
    MIN_PEAK_BRIGHT   = 200
    PERCENTILE_DARK   = 99.2
    PERCENTILE_BRIGHT = 97.5

    def detect(self,
               gray: np.ndarray,
               glint_roi: np.ndarray,
               is_dark: bool) -> List[GlintResult]:
        masked   = cv2.bitwise_and(gray, gray, mask=glint_roi)
        valid_px = masked[glint_roi > 0]
        if len(valid_px) == 0:
            return []

        pct = self.PERCENTILE_DARK if is_dark else self.PERCENTILE_BRIGHT
        thr = float(np.percentile(valid_px, pct))
        thr = max(thr, 150.0)

        _, thresh = cv2.threshold(masked, thr, 255, cv2.THRESH_BINARY)

        k = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, 8)

        glints: List[GlintResult] = []
        for lbl in range(1, n):
            area = float(stats[lbl, cv2.CC_STAT_AREA])
            if not (self.MIN_AREA < area < self.MAX_AREA):
                continue

            x0 = stats[lbl, cv2.CC_STAT_LEFT]
            y0 = stats[lbl, cv2.CC_STAT_TOP]
            bw = stats[lbl, cv2.CC_STAT_WIDTH]
            bh = stats[lbl, cv2.CC_STAT_HEIGHT]

            blob_mask = (labels[y0:y0+bh, x0:x0+bw] == lbl).astype(np.float64)
            blob_gray = gray[y0:y0+bh, x0:x0+bw].astype(np.float64)

            weights = blob_gray * blob_mask
            W = weights.sum()
            if W == 0:
                continue

            ys, xs = np.mgrid[y0:y0+bh, x0:x0+bw]
            cx = float((xs * weights).sum() / W)
            cy = float((ys * weights).sum() / W)

            blob_u8 = (labels == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(blob_u8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            perim = cv2.arcLength(cnts[0], True)
            circ  = min(1.0, (4.0 * math.pi * area / perim**2)
                        if perim > 0 else 0.0)

            if circ < self.MIN_CIRCULARITY:
                continue

            peak = float(blob_gray[blob_mask > 0].max())
            glints.append(GlintResult((cx, cy), area, peak, circ))

        peak_floor = self.MIN_PEAK_DARK if is_dark else self.MIN_PEAK_BRIGHT
        glints = [g for g in glints if g.peak_intensity >= peak_floor]
        glints.sort(key=lambda g: -g.peak_intensity)
        return glints

    @staticmethod
    def remove_from_image(gray: np.ndarray,
                          glints: List[GlintResult]) -> np.ndarray:
        if not glints:
            return gray.copy()
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for g in glints:
            ix, iy = int(round(g.center[0])), int(round(g.center[1]))
            r = int(max(4, math.sqrt(g.area / math.pi) * 3.0))
            cv2.circle(mask, (ix, iy), r, 255, -1)
        return cv2.inpaint(gray, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


# ─────────────────────────────────────────────────────────────────────────────
# KALMAN TRACKER  (new in v4)
# ─────────────────────────────────────────────────────────────────────────────

class PupilKalmanTracker:
    """
    5-state constant-velocity Kalman filter for pupil tracking.

    State  x = [px, py, r, vx, vy]
    Obs    z = [px, py, r]

    Process noise Q is tuned for human fixation (~0.5 px/frame jitter)
    with moderate velocity noise to allow for microsaccades and natural
    inter-target saccades (≤ 80 px/frame).

    Measurement noise R is adapted per-frame: inversely proportional to
    edge_support and proportional to fit_rms — worse detections contribute
    less to the state update.
    """

    def __init__(self) -> None:
        self._x: Optional[np.ndarray] = None    # [px, py, r, vx, vy]
        self._P: Optional[np.ndarray] = None    # 5×5 covariance

        dt = 1.0
        self._F = np.array([
            [1, 0, 0, dt,  0],
            [0, 1, 0,  0, dt],
            [0, 0, 1,  0,  0],
            [0, 0, 0,  1,  0],
            [0, 0, 0,  0,  1],
        ], dtype=np.float64)

        self._H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise: small for fixation, allows saccades via high vx/vy noise
        self._Q = np.diag([0.25, 0.25, 0.10, 16.0, 16.0]).astype(np.float64)

        # Base measurement noise (1 px std for position, 0.5 px for radius)
        self._R_base = np.diag([1.0, 1.0, 0.25]).astype(np.float64)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        return self._x is not None

    @property
    def position(self) -> Optional[Tuple[float, float]]:
        return (float(self._x[0]), float(self._x[1])) if self._x is not None else None

    @property
    def radius(self) -> Optional[float]:
        return float(self._x[2]) if self._x is not None else None

    # ── Core operations ───────────────────────────────────────────────────────

    def predict(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._x is None:
            return None, None
        x_p = self._F @ self._x
        P_p = self._F @ self._P @ self._F.T + self._Q
        return x_p, P_p

    def update(self,
               z: Tuple[float, float, float],
               fit_rms: float = 1.0,
               edge_support: float = 1.0
               ) -> Tuple[Optional[np.ndarray], float]:
        """
        Incorporate a new measurement.

        Returns
        -------
        (updated_state, mahalanobis_distance)
        mahalanobis_distance = 0.0 on first call (initialisation).
        """
        zv = np.array(z, dtype=np.float64)

        # ── Initialise on first detection ─────────────────────────────────
        if self._x is None:
            self._x = np.array([zv[0], zv[1], zv[2], 0.0, 0.0])
            self._P = np.diag([25.0, 25.0, 4.0, 400.0, 400.0]).astype(np.float64)
            return self._x.copy(), 0.0

        x_p, P_p = self.predict()

        # Adaptive measurement noise
        es   = max(0.05, min(1.0, edge_support))
        rms  = max(0.1, fit_rms)
        scale = (rms / 0.5) / es     # high rms or low es → large noise
        scale = min(scale, 25.0)     # cap to avoid filter going blind
        R = self._R_base * scale

        # Innovation
        y = zv - self._H @ x_p
        S = self._H @ P_p @ self._H.T + R

        try:
            S_inv  = np.linalg.inv(S)
            d_mah  = float(y @ S_inv @ y)
        except np.linalg.LinAlgError:
            d_mah = float('inf')
            S_inv = np.eye(3)

        # Kalman gain & update
        K       = P_p @ self._H.T @ S_inv
        self._x = x_p + K @ y
        self._P = (np.eye(5) - K @ self._H) @ P_p

        # Enforce physical bounds
        self._x[2] = max(5.0, self._x[2])   # radius ≥ 5 px

        return self._x.copy(), d_mah

    def reinitialize(self, x: float, y: float, r: float) -> None:
        """Reset tracker at a new fixation position (post-saccade)."""
        self._x = np.array([x, y, r, 0.0, 0.0])
        self._P = np.diag([16.0, 16.0, 4.0, 400.0, 400.0]).astype(np.float64)

    def reset(self) -> None:
        self._x = None
        self._P = None


# ─────────────────────────────────────────────────────────────────────────────
# S5 — PUPIL DETECTOR  (improved in v4)
# ─────────────────────────────────────────────────────────────────────────────

class PupilDetector:
    """
    Sub-pixel pupil detection: Canny edges + PROSAC + AMS ellipse fit.

    Changes from v2
    ---------------
    • Accepts optional (kalman_pos, kalman_r): when the Kalman tracker is
      initialised, the annular-band search is centred on the predicted position
      instead of the (potentially drifted) ROI centroid.  This is the single
      most impactful change for systematic-bias cases.
    • PROSAC sampling: edge points sorted by gradient magnitude; each RANSAC
      iteration draws 3 "elite" + 2 "standard" samples.
    • Radial gradient threshold raised to 0.50 (was 0.35) to reduce iris
      false-positives.
    • N_RANSAC = 100 (was 50).
    • INLIER_DIST = 1.5 px (was 2.0 px) — tighter inlier criterion enabled
      by better sampling.
    """

    R_MIN  = 7
    R_MAX  = 55     # hard cap: human pupil never exceeds ~55 px radius in 640×480
    AR_MIN = 0.40
    AR_MAX = 1.00

    MIN_EDGE_SUPPORT = 0.20
    MIN_INLIERS      = 8
    MAX_FIT_RMS      = 4.0

    N_RANSAC    = 100
    INLIER_DIST = 1.5

    CANNY_SIGMA_FACTOR_LOW  = 0.33
    CANNY_SIGMA_FACTOR_HIGH = 1.0

    RADIAL_THRESH_PRIMARY  = 0.50   # cos 60° — tighter than v2's cos 69°
    RADIAL_THRESH_FALLBACK = 0.35   # same as v2

    def __init__(self) -> None:
        self._rng = np.random.default_rng(seed=0)

    # ── Main entry point ──────────────────────────────────────────────────────

    def detect(self,
               gray_clean: np.ndarray,
               roi_tight: np.ndarray,
               roi_search: np.ndarray,
               is_dark: bool,
               r0_hint: Optional[float] = None,
               kalman_pos: Optional[Tuple[float, float]] = None,
               kalman_r: Optional[float] = None
               ) -> Optional[PupilResult]:
        """
        Parameters
        ----------
        gray_clean   : glint-free grayscale frame
        roi_tight    : tight pupil-core mask (temporal diff)
        roi_search   : dilated search mask (roi_tight + 40 px margin)
        is_dark      : True for dark frames
        r0_hint      : pre-CLOSE ROI radius (prevents over-inflated r0)
        kalman_pos   : Kalman-predicted pupil centre (x, y)  [v4 addition]
        kalman_r     : Kalman-predicted equivalent radius     [v4 addition]
        """
        h, w = gray_clean.shape

        # ── Step 1: Determine search centre and radius ────────────────────
        if kalman_pos is not None and kalman_r is not None:
            # v4: use Kalman prediction as primary seed
            init_center = kalman_pos
            init_radius = kalman_r
            # r0_hint is now the Kalman radius — use narrow bands by default
            use_tight_bands = True
        else:
            # Fall back to ROI-based estimate (v2 behaviour)
            init_center, init_radius = self._estimate_from_roi(roi_tight, h, w)
            if init_center is None:
                M = cv2.moments(roi_search)
                if M["m00"] > 0:
                    init_center = (float(M["m10"] / M["m00"]),
                                   float(M["m01"] / M["m00"]))
                    init_radius = 25.0
                else:
                    return None
            if (r0_hint is not None
                    and r0_hint > 3.0
                    and r0_hint < init_radius * 0.75):
                init_radius = r0_hint
            use_tight_bands = (r0_hint is not None and r0_hint > 5.0)

        # ── Step 2: CLAHE ─────────────────────────────────────────────────
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_clean)

        # ── Step 3: Sobel gradients ───────────────────────────────────────
        gx = cv2.Sobel(gray_clean, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_clean, cv2.CV_32F, 0, 1, ksize=3)

        # ── Step 4: Annular-band edge extraction ──────────────────────────
        if use_tight_bands or kalman_pos is not None:
            band_widths = (6, 10, 15, 25)
        else:
            band_widths = (15, 25)

        pts: Optional[np.ndarray] = None
        for band_w in band_widths:
            band_inner = float(max(3, init_radius - band_w))
            band_outer = float(init_radius + band_w)
            pts = self._annular_band_edges(
                gray_clean, enhanced, gx, gy,
                init_center, band_inner, band_outer,
                roi_search, is_dark
            )
            if len(pts) >= self.MIN_INLIERS:
                break

        if pts is None or len(pts) < self.MIN_INLIERS:
            return None

        # ── Step 5: PROSAC + AMS ─────────────────────────────────────────
        # Gradient-magnitude ordering for PROSAC (v4)
        gmag = self._gradient_magnitudes(pts, gx, gy)
        r_max = init_radius * 1.4

        ellipse = self._ransac_ams_prosac(pts, gmag,
                                          init_center=init_center,
                                          r_max=r_max)
        if ellipse is None:
            ellipse = self._fit_ams(pts)

        if ellipse is None or not self._ellipse_valid(ellipse, h, w):
            return None

        # ── Step 6: Quality metrics ───────────────────────────────────────
        valid_px = gray_clean[roi_search > 0]
        sigma    = float(np.std(valid_px)) if len(valid_px) > 0 else 30.0
        canny_lo = int(max(5,   sigma * self.CANNY_SIGMA_FACTOR_LOW))
        canny_hi = int(min(200, sigma * self.CANNY_SIGMA_FACTOR_HIGH))
        edges_full = cv2.Canny(gray_clean, canny_lo, canny_hi)
        edges_full = cv2.bitwise_and(edges_full, edges_full, mask=roi_search)

        edge_support = self._edge_support(ellipse, edges_full)
        inlier_mask  = self._inlier_mask(pts, ellipse)
        n_inliers    = int(inlier_mask.sum())
        fit_rms      = self._fit_rms(
            pts[inlier_mask] if n_inliers > 0 else pts, ellipse)

        if edge_support < self.MIN_EDGE_SUPPORT:
            return None
        if n_inliers < self.MIN_INLIERS:
            return None
        if fit_rms > self.MAX_FIT_RMS:
            return None

        (cx, cy), (MA, ma), _angle = ellipse
        a = MA / 2.0
        b = ma / 2.0
        return PupilResult(
            center       = (float(cx), float(cy)),
            ellipse      = ellipse,
            radius_eq    = math.sqrt(a * b),
            aspect_ratio = min(a, b) / max(a, b) if max(a, b) > 0 else 0.0,
            edge_support = edge_support,
            fit_rms      = fit_rms,
            n_inliers    = n_inliers,
        )

    # ── Gradient magnitudes at edge points (for PROSAC) ───────────────────

    @staticmethod
    def _gradient_magnitudes(pts: np.ndarray,
                              gx: np.ndarray,
                              gy: np.ndarray) -> np.ndarray:
        ex = np.clip(pts[:, 0].astype(int), 0, gx.shape[1] - 1)
        ey = np.clip(pts[:, 1].astype(int), 0, gx.shape[0] - 1)
        mag = np.sqrt(gx[ey, ex]**2 + gy[ey, ex]**2)
        return mag.astype(np.float32)

    # ── ROI seed ──────────────────────────────────────────────────────────────

    def _estimate_from_roi(self,
                           roi_tight: Optional[np.ndarray],
                           h: int, w: int
                           ) -> Tuple[Optional[Tuple[float, float]],
                                      Optional[float]]:
        if roi_tight is None:
            return None, None
        M = cv2.moments(roi_tight)
        if M["m00"] < 10:
            return None, None
        cx0   = float(M["m10"] / M["m00"])
        cy0   = float(M["m01"] / M["m00"])
        area0 = float(M["m00"])

        cnts, _ = cv2.findContours(roi_tight, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            sq  = cnt.squeeze()
            if sq.ndim == 2 and len(sq) >= 5:
                ell = self._fit_ams(sq.astype(np.float32))
                if ell is not None and self._ellipse_valid(ell, h, w):
                    (ex, ey), (MA, ma), _ = ell
                    cx0   = float(ex)
                    cy0   = float(ey)
                    area0 = math.pi * (MA / 2.0) * (ma / 2.0)

        r0 = float(math.sqrt(area0 / math.pi)) if area0 > 0 else 20.0
        return (cx0, cy0), r0

    # ── Annular-band Canny + gradient-direction filter ────────────────────────

    def _annular_band_edges(self,
                            gray_clean: np.ndarray,
                            enhanced: np.ndarray,
                            gx: np.ndarray,
                            gy: np.ndarray,
                            center: Tuple[float, float],
                            band_inner: float,
                            band_outer: float,
                            roi_mask: np.ndarray,
                            is_dark: bool) -> np.ndarray:
        cx, cy = center
        h, w   = gray_clean.shape

        Y, X  = np.ogrid[:h, :w]
        dist  = np.sqrt((X.astype(float) - cx)**2 + (Y.astype(float) - cy)**2)
        annular = ((dist >= band_inner) & (dist <= band_outer)).astype(np.uint8) * 255
        annular = cv2.bitwise_and(annular, roi_mask)

        if not annular.any():
            return np.empty((0, 2), dtype=np.float32)

        valid_px = enhanced[annular > 0]
        if len(valid_px) < 10:
            return np.empty((0, 2), dtype=np.float32)
        sigma_e = float(max(np.std(valid_px), 10.0))
        clo = int(max(5,   sigma_e * self.CANNY_SIGMA_FACTOR_LOW))
        chi = int(min(250, sigma_e * 1.2))
        edges = cv2.Canny(enhanced, clo, chi)
        edges = cv2.bitwise_and(edges, edges, mask=annular)

        ey, ex = np.where(edges > 0)
        if len(ex) == 0:
            return np.empty((0, 2), dtype=np.float32)

        rx = ex.astype(float) - cx
        ry = ey.astype(float) - cy
        r_mag = np.sqrt(rx**2 + ry**2)
        r_mag[r_mag < 1e-6] = 1.0
        rx /= r_mag;  ry /= r_mag

        grad_x = gx[ey, ex]
        grad_y = gy[ey, ex]
        g_mag  = np.sqrt(grad_x**2 + grad_y**2)
        g_mag[g_mag < 1e-6] = 1.0
        gnx = grad_x / g_mag
        gny = grad_y / g_mag

        dot = gnx * rx + gny * ry

        # v4: primary threshold 0.50 (was 0.35)
        if is_dark:
            keep = dot >= self.RADIAL_THRESH_PRIMARY
        else:
            keep = dot <= -self.RADIAL_THRESH_PRIMARY

        if keep.sum() < 8:
            # Fallback: accept any radially-aligned edge
            keep = np.abs(dot) >= self.RADIAL_THRESH_FALLBACK

        if not keep.any():
            return np.empty((0, 2), dtype=np.float32)

        return np.column_stack([ex[keep], ey[keep]]).astype(np.float32)

    # ── PROSAC RANSAC ─────────────────────────────────────────────────────────

    def _ransac_ams_prosac(self,
                           pts: np.ndarray,
                           gmag: np.ndarray,
                           init_center: Optional[Tuple[float, float]] = None,
                           r_max: Optional[float] = None
                           ) -> Optional[tuple]:
        """
        RANSAC with PROSAC-style sampling (gradient-magnitude ordering).

        Partition edge points into "elite" (top 25 % by gradient magnitude)
        and "standard" (remaining 75 %).  Each iteration samples 3 elite + 2
        standard, biasing toward high-gradient (more reliable) boundary points
        while still maintaining geometric diversity.

        Angular-stratification is applied within each tier so samples span
        the full ellipse perimeter.
        """
        n = len(pts)
        if n < 5:
            return None

        # Sort by descending gradient magnitude
        order     = np.argsort(-gmag)
        n_elite   = max(5, n // 4)            # top 25 %
        elite_idx = order[:n_elite]
        std_idx   = order[n_elite:]

        def _sector_split(idx_set):
            if init_center is None or len(idx_set) < 5:
                return None
            cx, cy = init_center
            angles = np.arctan2(pts[idx_set, 1] - cy, pts[idx_set, 0] - cx)
            order2 = np.argsort(angles)
            g      = len(idx_set) // 5
            return [idx_set[order2[i*g : (i+1)*g if i < 4 else len(idx_set)]]
                    for i in range(5) if g > 0 and
                    len(idx_set[order2[i*g : (i+1)*g if i < 4 else len(idx_set)]]) > 0]

        elite_sectors = _sector_split(elite_idx) if len(elite_idx) >= 5 else None
        std_sectors   = _sector_split(std_idx)   if len(std_idx)   >= 5 else None

        best_ellipse   = None
        best_n_inliers = 0

        for _ in range(self.N_RANSAC):
            # ── Sample 3 elite + 2 standard ───────────────────────────────
            if (elite_sectors is not None and len(elite_sectors) >= 3
                    and std_sectors is not None and len(std_sectors) >= 2):
                n_e = min(3, len(elite_sectors))
                n_s = min(2, len(std_sectors))
                e_idxs = self._rng.choice(len(elite_sectors), size=n_e, replace=False)
                s_idxs = self._rng.choice(len(std_sectors),   size=n_s, replace=False)
                sel_e = [int(self._rng.choice(elite_sectors[i])) for i in e_idxs]
                sel_s = [int(self._rng.choice(std_sectors[i]))   for i in s_idxs]
                sample = pts[sel_e + sel_s]
            elif init_center is not None:
                # Angular-stratified fallback
                cx, cy   = init_center
                angles   = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
                order_a  = np.argsort(angles)
                g        = n // 5
                sectors  = [order_a[i*g : (i+1)*g if i < 4 else n]
                            for i in range(5)]
                sectors  = [s for s in sectors if len(s) > 0]
                if len(sectors) >= 5:
                    selected = [self._rng.choice(s) for s in sectors[:5]]
                    sample   = pts[selected]
                else:
                    sample = pts[self._rng.choice(n, 5, replace=False)]
            else:
                sample = pts[self._rng.choice(n, 5, replace=False)]

            if len(sample) < 5:
                continue

            ell = self._fit_ams(sample)
            if ell is None:
                continue
            if not self._ellipse_valid_shape(ell):
                continue
            if r_max is not None:
                (_, _), (MA_c, ma_c), _ = ell
                if math.sqrt((MA_c / 2.0) * (ma_c / 2.0)) > r_max:
                    continue

            n_in = int(self._inlier_mask(pts, ell).sum())
            if n_in > best_n_inliers:
                best_n_inliers = n_in
                best_ellipse   = ell

        if best_ellipse is None or best_n_inliers < self.MIN_INLIERS:
            return None

        # Refit on all inliers
        inliers = pts[self._inlier_mask(pts, best_ellipse)]
        if len(inliers) < 5:
            return best_ellipse
        final = self._fit_ams(inliers)
        if final is not None:
            (_, _), (MA_f, ma_f), _ = final
            r_f = math.sqrt((MA_f / 2.0) * (ma_f / 2.0))
            if r_f > self.R_MAX:
                return best_ellipse  # refit exploded — keep RANSAC result
        return final if final is not None else best_ellipse

    # ── Shape validity ────────────────────────────────────────────────────────

    def _ellipse_valid_shape(self, ell: tuple) -> bool:
        (_, _), (MA, ma), _ = ell
        a, b = MA / 2.0, ma / 2.0
        if a <= 0 or b <= 0:
            return False
        r_eq = math.sqrt(a * b)
        if not (self.R_MIN <= r_eq <= self.R_MAX):
            return False
        return (min(a, b) / max(a, b)) >= self.AR_MIN

    def _ellipse_valid(self, ell: tuple, h: int, w: int) -> bool:
        (cx, cy), (MA, ma), _ = ell
        a, b = MA / 2.0, ma / 2.0
        if a <= 0 or b <= 0:
            return False
        r_eq = math.sqrt(a * b)
        if not (self.R_MIN <= r_eq <= self.R_MAX):
            return False
        ar = min(a, b) / max(a, b)
        if not (self.AR_MIN <= ar <= self.AR_MAX):
            return False
        margin = 3
        return (margin < cx < w - margin and margin < cy < h - margin)

    # ── Ellipse fitting ───────────────────────────────────────────────────────

    @staticmethod
    def _fit_ams(pts: np.ndarray) -> Optional[tuple]:
        if pts is None or len(pts) < 5:
            return None
        try:
            return cv2.fitEllipseAMS(pts)
        except Exception:
            pass
        try:
            return cv2.fitEllipse(pts)
        except Exception:
            return None

    # ── Point-to-ellipse distance (algebraic, for RANSAC speed) ───────────────

    @staticmethod
    def _pt_ellipse_dist(pts: np.ndarray, ell: tuple) -> np.ndarray:
        (cx, cy), (MA, ma), angle = ell
        a, b = MA / 2.0, ma / 2.0
        if a <= 0 or b <= 0:
            return np.full(len(pts), 1e9)
        theta = math.radians(angle)
        c, s  = math.cos(theta), math.sin(theta)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        xr = dx * c + dy * s
        yr = -dx * s + dy * c
        norm = np.sqrt((xr / a)**2 + (yr / b)**2)
        return np.abs(norm - 1.0) * min(a, b)

    def _inlier_mask(self, pts: np.ndarray, ell: tuple) -> np.ndarray:
        return self._pt_ellipse_dist(pts, ell) < self.INLIER_DIST

    def _fit_rms(self, pts: np.ndarray, ell: tuple) -> float:
        if len(pts) == 0:
            return 999.0
        d = self._pt_ellipse_dist(pts, ell)
        return float(np.sqrt(np.mean(d**2)))

    # ── Edge support ──────────────────────────────────────────────────────────

    @staticmethod
    def _edge_support(ell: tuple, edges: np.ndarray,
                       n_samples: int = 72, search_r: int = 3) -> float:
        (cx, cy), (MA, ma), angle = ell
        a, b = MA / 2.0, ma / 2.0
        h, w = edges.shape
        theta = math.radians(angle)
        c, s  = math.cos(theta), math.sin(theta)
        hit = valid = 0
        for i in range(n_samples):
            t  = 2.0 * math.pi * i / n_samples
            xe = a * math.cos(t)
            ye = b * math.sin(t)
            xr = xe * c - ye * s + cx
            yr = xe * s + ye * c + cy
            xi, yi = int(round(xr)), int(round(yr))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            valid += 1
            r  = search_r
            x1, x2 = max(0, xi - r), min(w, xi + r + 1)
            y1, y2 = max(0, yi - r), min(h, yi + r + 1)
            if edges[y1:y2, x1:x2].any():
                hit += 1
        return hit / valid if valid > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GLINT SELECTOR  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

class GlintSelector:
    MAX_DIST = 150

    @staticmethod
    def for_dark(pupil_center: Tuple[float, float],
                 glints: List[GlintResult]) -> List[GlintResult]:
        px, py   = pupil_center
        filtered = [g for g in glints
                    if math.hypot(g.center[0]-px, g.center[1]-py) < GlintSelector.MAX_DIST]
        filtered.sort(key=lambda g: math.hypot(g.center[0]-px, g.center[1]-py))
        return filtered[:2]

    @staticmethod
    def for_bright(pupil_center: Tuple[float, float],
                   glints: List[GlintResult],
                   pupil_radius: float) -> List[GlintResult]:
        px, py = pupil_center
        scored = []
        for g in glints:
            dist = math.hypot(g.center[0]-px, g.center[1]-py)
            if dist > GlintSelector.MAX_DIST:
                continue
            proximity_score = 1.0 - dist / GlintSelector.MAX_DIST
            score = g.circularity * 0.5 + proximity_score * 0.5
            scored.append((score, g))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[0])
        return [scored[0][1]]

    @staticmethod
    def midpoint(g1: GlintResult, g2: GlintResult) -> Tuple[float, float]:
        return ((g1.center[0] + g2.center[0]) / 2.0,
                (g1.center[1] + g2.center[1]) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# S6 — BLINK / QUALITY DETECTOR  (identical to v2)
# ─────────────────────────────────────────────────────────────────────────────

class BlinkDetector:
    HISTORY      = 30
    MIN_AR       = 0.30
    MIN_ES       = 0.15
    MAX_FIT_RMS  = 6.0
    MAX_R_JUMP   = 2.5

    def __init__(self) -> None:
        self._radius_hist: deque = deque(maxlen=self.HISTORY)

    def evaluate(self, pupil: Optional[PupilResult]) -> Tuple[bool, str]:
        if pupil is None:
            return True, "B1:no_pupil"
        if pupil.aspect_ratio < self.MIN_AR:
            return True, f"B2:ar={pupil.aspect_ratio:.2f}"
        if pupil.edge_support < self.MIN_ES:
            return True, f"B3:es={pupil.edge_support:.2f}"
        if pupil.fit_rms > self.MAX_FIT_RMS:
            return True, f"B4:rms={pupil.fit_rms:.2f}"
        if len(self._radius_hist) >= 5:
            med = float(np.median(list(self._radius_hist)))
            if med > 0:
                ratio = pupil.radius_eq / med
                if ratio < 1.0 / self.MAX_R_JUMP or ratio > self.MAX_R_JUMP:
                    return True, f"B5:r_jump={ratio:.2f}"
        self._radius_hist.append(pupil.radius_eq)
        return False, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class EyeTrackingPipeline:
    """
    Orchestrates the full v4 pipeline, integrating the Kalman tracker.

    Kalman integration logic
    ------------------------
    (a) If tracker is initialised:
        - Pass Kalman prediction to PupilDetector as search centre.
        - After detection, compute Mahalanobis distance.
        - If d < _CHI2_999: normal update (fixation).
        - If d >= _CHI2_999 AND quality good AND dist < _SACCADE_DIST_PX:
          saccade → reinitialise tracker at new position.
        - If d >= _CHI2_999 AND quality poor: outlier → mark as blink.

    (b) If tracker is not initialised:
        - Behave exactly like v2 until first valid detection.

    (c) Synthetic ROI from Kalman:
        - When the temporal-diff ROI is None but the tracker is initialised,
          create a circular mask at the predicted position so detection can
          continue across long sequences of same-type frames.
    """

    def __init__(self) -> None:
        self.classifier = FrameClassifier()
        self.roi_est    = ROIEstimator()
        self.glint_det  = GlintDetector()
        self.pupil_det  = PupilDetector()
        self.blink_det  = BlinkDetector()
        self.kalman     = PupilKalmanTracker()

        self._buffer: deque = deque(maxlen=3)
        self._n_total = 0
        self._n_valid = 0

    # ── Per-frame processing ──────────────────────────────────────────────────

    def process(self, gray: np.ndarray,
                filename: str,
                frame_idx: int) -> FrameResult:

        ftype = self.classifier.classify(gray)
        self._buffer.append({"gray": gray, "type": ftype,
                              "filename": filename, "idx": frame_idx})
        self._n_total += 1

        result = FrameResult(filename=filename, frame_idx=frame_idx,
                             frame_type=ftype, is_blink=True)

        # ── ROI (temporal diff or Kalman synthetic) ───────────────────────
        roi_tight         = self._get_roi()
        using_synth_roi   = False

        if roi_tight is None and self.kalman.is_initialized:
            roi_tight       = self._synthetic_roi(self.kalman.position,
                                                   self.kalman.radius, gray.shape)
            using_synth_roi = True

        if roi_tight is None:
            result.blink_reason = "no_roi"
            return result

        roi_pupil = self.roi_est.dilate_for_pupil(roi_tight, px=40)
        roi_glint = self.roi_est.dilate_for_glints(roi_tight, px=65)

        # ── Glint detection (on original gray) ───────────────────────────
        is_dark    = (ftype == "dark")
        raw_glints = self.glint_det.detect(gray, roi_glint, is_dark)
        gray_clean = GlintDetector.remove_from_image(gray, raw_glints)

        # ── Pupil detection ────────────────────────────────────────────────
        # Kalman position is used as band centre ONLY when no temporal-diff ROI
        # is available (synthetic ROI mode).  When a real ROI exists, the ROI
        # centroid is used — this prevents the Kalman from introducing a
        # systematic downward bias that grows via positive feedback.
        kpos = self.kalman.position
        krad = self.kalman.radius

        pupil = self.pupil_det.detect(
            gray_clean, roi_tight, roi_pupil, is_dark,
            r0_hint   = self.roi_est._last_r0_hint,
            kalman_pos= kpos if using_synth_roi else None,
            kalman_r  = krad if using_synth_roi else None,
        )

        # ── Kalman update / outlier gate ──────────────────────────────────
        if pupil is not None:
            meas = (pupil.center[0], pupil.center[1], pupil.radius_eq)
            _, d_mah = self.kalman.update(meas, pupil.fit_rms, pupil.edge_support)

            if kpos is not None and d_mah >= _CHI2_999:
                dist_to_pred = math.hypot(
                    pupil.center[0] - kpos[0],
                    pupil.center[1] - kpos[1])

                good_quality = (pupil.edge_support >= 0.35
                                and pupil.fit_rms <= 3.5)

                if good_quality and dist_to_pred <= _SACCADE_DIST_PX:
                    # Saccade: reinitialise Kalman and clear stale ROI history
                    # so the next frames use a fresh synthetic ROI centred at
                    # the new fixation position rather than the old one.
                    self.kalman.reinitialize(*meas)
                    self.roi_est._history.clear()
                    self.roi_est._last_valid = None
                else:
                    # Outlier: reject detection, propagate Kalman prediction
                    result.blink_reason = "K:outlier"
                    return result

        # ── Blink detection ───────────────────────────────────────────────
        is_blink, reason = self.blink_det.evaluate(pupil)
        result.is_blink     = is_blink
        result.blink_reason = reason

        if is_blink or pupil is None:
            return result

        result.pupil = pupil

        # ── Glint selection ───────────────────────────────────────────────
        if is_dark:
            sel = GlintSelector.for_dark(pupil.center, raw_glints)
        else:
            sel = GlintSelector.for_bright(pupil.center, raw_glints,
                                            pupil.radius_eq)
        result.glints = sel

        if is_dark:
            if len(sel) >= 2:
                result.glint_ref = GlintSelector.midpoint(sel[0], sel[1])
            elif len(sel) == 1:
                result.glint_ref = sel[0].center
        elif sel:
            result.glint_ref = sel[0].center

        if result.glint_ref is not None:
            result.vector = (
                pupil.center[0] - result.glint_ref[0],
                pupil.center[1] - result.glint_ref[1],
            )

        result.quality = pupil.edge_support
        self._n_valid += 1
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_roi(self) -> Optional[np.ndarray]:
        buf = list(self._buffer)
        if len(buf) >= 2:
            f1, f2 = buf[-2], buf[-1]
            if (f1["type"] != f2["type"]
                    and "unknown" not in (f1["type"], f2["type"])):
                dark_f   = f1 if f1["type"] == "dark"   else f2
                bright_f = f1 if f1["type"] == "bright" else f2
                roi = self.roi_est.estimate_from_pair(dark_f["gray"],
                                                       bright_f["gray"])
                if roi is not None:
                    return roi
        return self.roi_est.get_fallback()

    @staticmethod
    def _synthetic_roi(pos: Tuple[float, float],
                       radius: Optional[float],
                       shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Create a circular binary mask centred at the Kalman prediction."""
        if pos is None:
            return None
        h, w = shape
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        r = int(max(15, min(60, (radius or 25) * 1.5)))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask

    def stats(self) -> Dict:
        return {
            "total":      self._n_total,
            "valid":      self._n_valid,
            "valid_rate": self._n_valid / max(1, self._n_total),
        }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

C_PUPIL_CROSS = (255,  80,   0)
C_ELLIPSE     = (  0, 230,   0)
C_GLINT_1     = (  0,   0, 255)
C_GLINT_2     = (  0, 200, 255)
C_GLINT_REF   = (255, 255,   0)
C_TEXT        = (255, 255, 255)
C_BLINK       = (  0,   0, 255)


def draw_detection(bgr: np.ndarray, result: FrameResult,
                   roi_mask: Optional[np.ndarray] = None,
                   diff_img: Optional[np.ndarray] = None,
                   edges_img: Optional[np.ndarray] = None) -> np.ndarray:
    h, w = bgr.shape[:2]
    A = bgr.copy()
    if result.is_blink:
        cv2.putText(A, f"BLINK  {result.blink_reason}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BLINK, 2)
    else:
        if result.pupil:
            cx, cy = result.pupil.center
            cv2.line(A, (0, int(cy)), (w, int(cy)), C_PUPIL_CROSS, 1)
            cv2.line(A, (int(cx), 0), (int(cx), h), C_PUPIL_CROSS, 1)
            cv2.circle(A, (int(round(cx)), int(round(cy))), 3, C_PUPIL_CROSS, -1)
            cv2.ellipse(A, result.pupil.ellipse, C_ELLIPSE, 1)
        for i, g in enumerate(result.glints):
            gx, gy = g.center
            col = C_GLINT_1 if i == 0 else C_GLINT_2
            cv2.line(A, (0, int(gy)), (w, int(gy)), col, 1)
            cv2.line(A, (int(gx), 0), (int(gx), h), col, 1)
            cv2.circle(A, (int(round(gx)), int(round(gy))), 4, col, -1)
        if result.glint_ref:
            rx, ry = result.glint_ref
            cv2.circle(A, (int(round(rx)), int(round(ry))), 4, C_GLINT_REF, 1)
        if result.pupil:
            txt = (f"F{result.frame_idx} {result.frame_type.upper()}  "
                   f"P=({result.pupil.center[0]:.1f},{result.pupil.center[1]:.1f})  "
                   f"R={result.pupil.radius_eq:.1f}px  "
                   f"ES={result.pupil.edge_support:.2f}  "
                   f"RMS={result.pupil.fit_rms:.2f}")
            cv2.putText(A, txt, (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT, 1, cv2.LINE_AA)

    B = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)  if roi_mask  is not None else np.zeros_like(A)
    C = cv2.cvtColor(cv2.equalizeHist(diff_img),
                     cv2.COLOR_GRAY2BGR)             if diff_img  is not None else np.zeros_like(A)
    D = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)  if edges_img is not None else np.zeros_like(A)

    if roi_mask  is not None: cv2.putText(B, "ROI mask",    (5,18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    if diff_img  is not None: cv2.putText(C, "absdiff(eq)", (5,18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
    if edges_img is not None: cv2.putText(D, "Canny edges", (5,18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)

    return np.vstack([np.hstack([A, B]), np.hstack([C, D])])


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT  (identical format to v2 — drop-in compatible)
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(results: List[FrameResult],
               output_path: str,
               input_csv_path: Optional[str] = None) -> None:
    rows = {}
    for r in results:
        px, py   = r.pupil.center   if r.pupil       else (0.0, 0.0)
        gx, gy   = r.glint_ref      if r.glint_ref   else (0.0, 0.0)
        vx, vy   = r.vector         if r.vector      else (0.0, 0.0)
        g1x, g1y = r.glints[0].center if len(r.glints) > 0 else (0.0, 0.0)
        g2x, g2y = r.glints[1].center if len(r.glints) > 1 else (0.0, 0.0)
        rows[r.filename] = {
            "status":       0 if r.is_blink else 1,
            "pupil_x":      round(px, 4),
            "pupil_y":      round(py, 4),
            "glint_x":      round(gx, 4),
            "glint_y":      round(gy, 4),
            "glint1_x":     round(g1x, 4),
            "glint1_y":     round(g1y, 4),
            "glint2_x":     round(g2x, 4),
            "glint2_y":     round(g2y, 4),
            "vector_x":     round(vx, 4),
            "vector_y":     round(vy, 4),
            "frame_type":   r.frame_type,
            "quality":      round(r.quality, 4),
            "edge_support": round(r.pupil.edge_support, 4) if r.pupil else 0.0,
            "fit_rms":      round(r.pupil.fit_rms, 4)      if r.pupil else 0.0,
            "pupil_radius": round(r.pupil.radius_eq, 4)    if r.pupil else 0.0,
            "aspect_ratio": round(r.pupil.aspect_ratio, 4) if r.pupil else 0.0,
            "n_inliers":    r.pupil.n_inliers               if r.pupil else 0,
            "blink_reason": r.blink_reason,
        }

    if input_csv_path and os.path.exists(input_csv_path):
        try:
            df = pd.read_csv(input_csv_path)
            if any(";" in str(c) for c in df.columns):
                df = pd.read_csv(input_csv_path, sep=";")
            df.columns = [c.strip() for c in df.columns]
            for alias in ("file_name", "filename", "FILE_NAME", "image", "frame"):
                if alias in df.columns:
                    if alias != "file_name":
                        df.rename(columns={alias: "file_name"}, inplace=True)
                    break
            sample = next(iter(rows.values())) if rows else {}
            for col, default in sample.items():
                if col not in df.columns:
                    df[col] = type(default)()
            for fname, data in rows.items():
                mask = df["file_name"] == fname
                if mask.any():
                    for col, val in data.items():
                        df.loc[mask, col] = val
            df.to_csv(output_path, index=False)
            print(f"[CSV] Merged → {output_path}")
            return
        except Exception as exc:
            print(f"[WARN] Could not merge: {exc}")

    pd.DataFrame([{"file_name": fn, **data} for fn, data in rows.items()]
                 ).to_csv(output_path, index=False)
    print(f"[CSV] → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(input_folder: str,
         output_folder: str,
         show_gui: bool = True,
         auto_mode: bool = False,
         save_debug_images: bool = False) -> None:

    VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".pgm")
    files = sorted(f for f in os.listdir(input_folder)
                   if f.lower().endswith(VALID_EXT))
    if not files:
        print(f"[ERROR] No images in {input_folder}")
        return

    total = len(files)
    print(f"[i] {total} images found in: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    csvs      = [f for f in os.listdir(input_folder)
                 if f.lower().endswith(".csv") and "processed" not in f.lower()]
    input_csv = os.path.join(input_folder, csvs[0]) if csvs else None

    pipeline  = EyeTrackingPipeline()
    results: List[FrameResult] = []

    # ── v4 addition: always prescan before processing ─────────────────────
    print("[i] Prescanning frames for optimal dark/bright threshold …")
    all_counts = []
    for fn in files:
        g = cv2.imread(os.path.join(input_folder, fn), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            all_counts.append(pipeline.classifier.bright_pixel_count(g))
    pipeline.classifier.prescan(all_counts)
    print(f"[i] Prescan complete. Threshold = "
          f"{pipeline.classifier._adaptive_thresh:.1f} bright-px")

    WIN = "PupilGlint v4 | [N]=Next  [A]=Auto  [Q]=Quit"
    if show_gui:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, NATIVE_W * 2, NATIVE_H * 2)

    _last_roi: Optional[np.ndarray]   = None
    _last_diff: Optional[np.ndarray]  = None
    _last_edges: Optional[np.ndarray] = None

    for idx, filename in enumerate(files):
        path = os.path.join(input_folder, filename)
        bgr  = cv2.imread(path)
        if bgr is None:
            continue
        if bgr.shape[:2] != (NATIVE_H, NATIVE_W):
            bgr = cv2.resize(bgr, (NATIVE_W, NATIVE_H))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        result = pipeline.process(gray, filename, idx + 1)
        results.append(result)

        if pipeline.roi_est._last_valid is not None:
            _last_roi = pipeline.roi_est._last_valid

        buf = list(pipeline._buffer)
        if len(buf) >= 2 and buf[-2]["type"] != buf[-1]["type"]:
            dk = buf[-2]["gray"] if buf[-2]["type"] == "dark"   else buf[-1]["gray"]
            br = buf[-2]["gray"] if buf[-2]["type"] == "bright" else buf[-1]["gray"]
            _last_diff = cv2.absdiff(dk, br)

        if result.pupil and not result.is_blink and _last_roi is not None:
            clean = GlintDetector.remove_from_image(gray, result.glints)
            valid = clean[clean > 0] if np.any(clean > 0) else np.array([30])
            sig   = float(np.std(valid)) if len(valid) else 30.0
            clo   = int(max(5,   sig * PupilDetector.CANNY_SIGMA_FACTOR_LOW))
            chi   = int(min(200, sig * PupilDetector.CANNY_SIGMA_FACTOR_HIGH))
            _last_edges = cv2.bitwise_and(cv2.Canny(clean, clo, chi),
                                          cv2.Canny(clean, clo, chi),
                                          mask=_last_roi)

        if not auto_mode:
            if result.is_blink:
                print(f"[F{idx+1:05d}] {filename}  {result.frame_type.upper():7s}  "
                      f"BLINK ({result.blink_reason})")
            else:
                cx, cy = result.pupil.center
                print(f"[F{idx+1:05d}] {filename}  {result.frame_type.upper():7s}  "
                      f"P=({cx:.2f},{cy:.2f})  R={result.pupil.radius_eq:.1f}  "
                      f"ES={result.pupil.edge_support:.2f}  "
                      f"RMS={result.pupil.fit_rms:.2f}  G={len(result.glints)}")
        elif (idx + 1) % 50 == 0:
            st = pipeline.stats()
            print(f"[i] {idx+1}/{total}  valid={st['valid_rate']:.1%}")

        if show_gui:
            mosaic = draw_detection(bgr, result,
                                    roi_mask  = _last_roi,
                                    diff_img  = _last_diff,
                                    edges_img = _last_edges)
            if save_debug_images:
                dbg_dir = os.path.join(output_folder, "debug_frames")
                os.makedirs(dbg_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dbg_dir,
                            f"{os.path.splitext(filename)[0]}_debug.png"), mosaic)
            cv2.imshow(WIN, mosaic)
            key = cv2.waitKey(1 if auto_mode else 0) & 0xFF
            if key == ord("q"):
                print("[i] User quit.")
                break
            elif key == ord("a"):
                auto_mode = True

    cv2.destroyAllWindows()

    folder_name  = os.path.basename(os.path.normpath(input_folder))
    csv_out_path = os.path.join(output_folder, f"{folder_name}_segmentation_v4.csv")
    export_csv(results, csv_out_path, input_csv)

    st = pipeline.stats()
    print(f"\n[DONE] Processed: {st['total']}  Valid: {st['valid']} ({st['valid_rate']:.1%})")
    print(f"[DONE] CSV: {csv_out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sub-pixel pupil & glint detector v4 — stroboscopic P-CR"
    )
    p.add_argument("--input_folder",  "-i", required=False,
                   default=r"data/P10/fp1")
    p.add_argument("--output_folder", "-o", required=False,
                   default=r"data/resultado_pupila_glint/P10_v4")
    p.add_argument("--auto",       action="store_true", default=False)
    p.add_argument("--no_gui",     action="store_true", default=False)
    p.add_argument("--save_debug", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        input_folder      = args.input_folder,
        output_folder     = args.output_folder,
        show_gui          = not args.no_gui,
        auto_mode         = args.auto,
        save_debug_images = args.save_debug,
    )
