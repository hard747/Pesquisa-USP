#!/usr/bin/env python3
"""
Pipeline 15 Usuarios — Sessão 1  (Chin Rest, Sem Normalização T18)
====================================================================
Processa exatamente os 15 sujeitos na ordem definida, apenas sessões
fp1 fq1 fr1 fs1 ft1 fu1 fv1 (sessão 1), unificando todos os dados sem
normalização cross-session porque todos os sujeitos usaram chin rest.

Etapas:
  1. Segmentação (pupil_glint_detector_v2) — com cache por sessão
  2. Calibração: Polynomial Ridge grau 2, 9 alvos de treino, 26 de teste
  3. Filtro Exponencial (EMA) nos vetores de gaze estimado — α otimizado
  4. Exporta duas planilhas Excel por sujeito:
       tabela_segmentacao.xlsx   — dados de detecção (pupila, glints, vetor)
       tabela_estimacao.xlsx     — gaze estimado + suavizado
  5. Relatório final CSV + mapa de calor agregado

Uso:
    python3 pipeline_15_participantes.py              # todos os 15
    python3 pipeline_15_participantes.py --only P01 P02  # seleção
    python3 pipeline_15_participantes.py --force       # re-segmenta tudo
"""

import sys, os, math, argparse, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import RBFInterpolator
from scipy.signal import savgol_filter

# ── importa pipeline de segmentação v2 ───────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from pupil_glint_detector_v2 import (
    EyeTrackingPipeline, export_csv, FrameResult,
    NATIVE_W, NATIVE_H,
)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR   = "data"
OUTPUT_DIR = "results_15usuarios"

# Ordem dos 15 sujeitos conforme solicitado
USERS = [
    "P01", "P02", "P03", "P04", "P05",
    "P06", "P07", "P08", "P09", "P10",
    "P11", "P12", "P13", "P14", "P15",
]

# Sujeitos com estrutura de diretório aninhada: data/<user>/<subdir>/fp1/
NESTED = {
    "P04":  "P04_fixacoes",
    "P08":   "P08_fixacoes",
    "P13": "P13_fixacoes",
}

# Sessões da rodada 1 apenas (grupos de alvos não se repetem entre sessões)
SESSION1 = ["fp1", "fq1", "fr1", "fs1", "ft1", "fu1", "fv1"]

VALID_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".pgm", ".tif")

# ── Geometria do experimento ──────────────────────────────────────────────────
SCREEN_W_PX   = 1920
SCREEN_H_PX   = 1080
SCREEN_W_MM   = 507.0
DISTANCE_MM   = 600.0
PIXEL_PITCH   = SCREEN_W_MM / SCREEN_W_PX           # 0.2641 mm/px
ONE_DEG_PX    = DISTANCE_MM * math.tan(math.radians(1)) / PIXEL_PITCH  # ≈39.7

# ── Grade de alvos 5×7 ───────────────────────────────────────────────────────
COLS = [135, 410, 685, 960, 1235, 1510, 1785]
ROWS = [108, 324, 540, 756, 972]

# 9 alvos para TREINO do polinômio (grade 3×3 dispersa)
CALIB_IDS = {1, 4, 7, 15, 18, 21, 29, 32, 35}

# ── Parâmetros de calibração ──────────────────────────────────────────────────
POLY_DEGREE  = 2
RIDGE_ALPHA  = 5.0
MIN_SEG_FRAMES  = 3   # mínimo de frames por segmento (depois de IQR)
MIN_TGT_FRAMES  = 5   # mínimo de frames acumulados por alvo
MAX_GLINT_DIST  = 60  # px — rejeita glints obviamente errados

# ── Filtro EMA ────────────────────────────────────────────────────────────────
EMA_ALPHAS    = [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
TARGET_ERROR  = 2.0   # ° — meta de erro após EMA


# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════════════════════════════════════════════

def _tid(x: float, y: float) -> int:
    try:
        return ROWS.index(int(round(y))) * 7 + COLS.index(int(round(x))) + 1
    except ValueError:
        return -1


def _deg(error_px: float) -> float:
    return math.degrees(math.atan(abs(error_px) * PIXEL_PITCH / DISTANCE_MM))


def _vec_deg(ex: float, ey: float) -> float:
    return math.degrees(math.atan(math.hypot(ex, ey) * PIXEL_PITCH / DISTANCE_MM))


def find_session_dir(data_dir: str, user: str, session: str) -> Optional[str]:
    """
    Resolve o caminho real da sessão levando em conta a estrutura aninhada
    de P04 / P08 / P13.
    """
    if user in NESTED:
        path = os.path.join(data_dir, user, NESTED[user], session)
    else:
        path = os.path.join(data_dir, user, session)
    return path if os.path.isdir(path) else None


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — SEGMENTAÇÃO
# ═════════════════════════════════════════════════════════════════════════════

def run_segmentation(session_dir: str, out_csv: str, force: bool = False) -> Optional[str]:
    """
    Roda EyeTrackingPipeline v2 em todos os frames da sessão e exporta CSV.
    Com cache: se o CSV já existe e force=False, pula o processamento.
    """
    sess_name = os.path.basename(session_dir)

    if os.path.exists(out_csv) and not force:
        try:
            df = pd.read_csv(out_csv)
            n_v = int((df["status"] == 1).sum())
            print(f"      [cache] {sess_name}: {n_v}/{len(df)} ({n_v/len(df):.1%})")
        except Exception:
            print(f"      [cache] {sess_name}")
        return out_csv

    imgs = sorted(
        f for f in os.listdir(session_dir) if f.lower().endswith(VALID_IMG_EXT)
    )
    if not imgs:
        print(f"      [SKIP]  {sess_name}: sem imagens")
        return None

    # Coleta CSV da sessão original (para timestamps e coordenadas de alvo)
    coleta_csvs = [
        f for f in os.listdir(session_dir)
        if f.lower().endswith(".csv")
        and "processed" not in f.lower()
        and "segmentation" not in f.lower()
    ]
    input_csv = os.path.join(session_dir, coleta_csvs[0]) if coleta_csvs else None

    pipeline = EyeTrackingPipeline()

    # ── Prescan: compute global threshold for frame classifier ──────────
    # Offline mode: scan all frames once to compute n_bright counts, then
    # set the optimal dark/bright threshold before processing begins.
    # This avoids the rolling-Otsu bug on sparse arrays (10 values).
    prescan_counts = []
    for fname in imgs:
        bgr = cv2.imread(os.path.join(session_dir, fname))
        if bgr is None:
            continue
        if bgr.shape[:2] != (NATIVE_H, NATIVE_W):
            bgr = cv2.resize(bgr, (NATIVE_W, NATIVE_H))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        prescan_counts.append(pipeline.classifier.bright_pixel_count(gray))
    if prescan_counts:
        pipeline.classifier.prescan(prescan_counts)

    results: List[FrameResult] = []
    t0 = time.time()

    for idx, fname in enumerate(imgs):
        bgr = cv2.imread(os.path.join(session_dir, fname))
        if bgr is None:
            continue
        if bgr.shape[:2] != (NATIVE_H, NATIVE_W):
            bgr = cv2.resize(bgr, (NATIVE_W, NATIVE_H))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        results.append(pipeline.process(gray, fname, idx + 1))

    if not results:
        print(f"      [FAIL]  {sess_name}: nenhum frame")
        return None

    export_csv(results, out_csv, input_csv)
    st = pipeline.stats()
    fps = len(imgs) / (time.time() - t0)
    print(f"      {sess_name}: {st['valid']}/{st['total']} "
          f"({st['valid_rate']:.1%})   {fps:.0f} fps")
    return out_csv


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — CARREGA E AGREGA DADOS (SEM NORMALIZAÇÃO T18)
# ═════════════════════════════════════════════════════════════════════════════

def _iqr_filter(seg: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers IQR em vector_x e vector_y."""
    for col in ("vector_x", "vector_y"):
        q1, q3 = seg[col].quantile(0.25), seg[col].quantile(0.75)
        iqr = q3 - q1
        seg = seg[(seg[col] >= q1 - 1.5 * iqr) & (seg[col] <= q3 + 1.5 * iqr)]
    return seg


def load_sessions_no_norm(csv_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lê todos os CSVs de sessão-1 SEM normalização por T18 (chin rest →
    posição da cabeça estável entre sessões → vetores diretamente comparáveis).

    Retorna dois DataFrames:
        df_calib  — um row por alvo, com vetor médio (para calibração)
        df_frames — todos os frames dark válidos (para tabela Excel e EMA)
    """
    needed = {"timestamp", "status", "vector_x", "vector_y",
              "tar1_x", "tar1_y", "pupil_x", "pupil_y",
              "glint_x", "glint_y", "frame_type",
              "pupil_radius", "aspect_ratio", "edge_support",
              "fit_rms", "n_inliers", "quality"}

    all_rows  = []   # para calibração (agregação por alvo)
    all_dark  = []   # todos os frames dark válidos (para EMA e Excel)

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # Aceita CSVs mesmo sem todas as colunas opcionais de qualidade
        if not {"timestamp","status","vector_x","vector_y",
                "tar1_x","tar1_y","frame_type"}.issubset(df.columns):
            continue

        # Filtros de qualidade: apenas dark frames, com glint (vetor ≠ 0)
        df_dark = df[(df["status"] == 1) &
                     (df["frame_type"] == "dark") &
                     (df["vector_x"] != 0)].copy()

        # Rejeita glints >MAX_GLINT_DIST px do centro da pupila
        if {"pupil_x","pupil_y","glint_x","glint_y"}.issubset(df.columns):
            gd = np.sqrt((df_dark["pupil_x"] - df_dark["glint_x"])**2 +
                         (df_dark["pupil_y"] - df_dark["glint_y"])**2)
            df_dark = df_dark[gd <= MAX_GLINT_DIST]

        if df_dark.empty:
            continue

        # Guarda frames com nome da sessão para Excel
        sess_name = os.path.splitext(os.path.basename(path))[0]
        sess_name = sess_name.replace("_segmentation_v2", "")
        df_dark = df_dark.copy()
        df_dark["session"] = sess_name
        all_dark.append(df_dark)

        # Segmenta por alvo e calcula vetor médio por segmento
        df_dark["_chg"] = (
            (df_dark["tar1_x"] != df_dark["tar1_x"].shift()) |
            (df_dark["tar1_y"] != df_dark["tar1_y"].shift())
        )
        df_dark["_seg"] = df_dark["_chg"].cumsum()

        for _, grp in df_dark.groupby("_seg", sort=False):
            tx = float(grp["tar1_x"].iloc[0])
            ty = float(grp["tar1_y"].iloc[0])
            if tx <= 0 or ty <= 0:
                continue
            tid = _tid(tx, ty)
            if tid == -1:
                continue
            seg = _iqr_filter(grp)
            if len(seg) < MIN_SEG_FRAMES:
                continue
            all_rows.append({
                "target_id": tid,
                "target_x":  tx,
                "target_y":  ty,
                "vector_x":  float(seg["vector_x"].mean()),
                "vector_y":  float(seg["vector_y"].mean()),
                "count":     len(seg),
            })

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    # Agrega por alvo (média dos segmentos de cada alvo)
    df_all = pd.DataFrame(all_rows)
    df_calib = df_all.groupby("target_id").agg(
        vector_x=("vector_x", "mean"),
        vector_y=("vector_y", "mean"),
        target_x=("target_x", "first"),
        target_y=("target_y", "first"),
        count=("count",    "sum"),
    ).reset_index()
    df_calib = df_calib[df_calib["count"] >= MIN_TGT_FRAMES].reset_index(drop=True)

    df_frames = pd.concat(all_dark, ignore_index=True) if all_dark else pd.DataFrame()
    return df_calib, df_frames


def load_sessions_pupil_corrected(csv_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calibração baseada em POSIÇÃO DA PUPILA (não vetor P-CR).

    Com chin rest a cabeça fica estável →  a posição da pupila na imagem é
    diretamente proporcional à direção do olhar (sensibilidade ~7.4 px/°,
    vs ~0.85 px/° do vetor P-CR).

    Correção de drift da câmera entre sessões: subtrai o centro médio da
    pupila em T18 (960,540) de cada sessão.  T18 aparece em TODAS as 7
    sessões, por isso é o ponto de referência perfeito.

    Colunas de saída: 'feat_x','feat_y' = posição da pupila corrigida.
    """
    T18X, T18Y = 960, 540

    # Passo 1 — posição média da pupila em T18 por sessão
    t18_pupil_per_sess: Dict[str, Tuple[float,float]] = {}
    raw_dark_per_sess:  Dict[str, pd.DataFrame]       = {}

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"timestamp","status","pupil_x","pupil_y",
                "tar1_x","tar1_y","frame_type"}.issubset(df.columns):
            continue

        df_dark = df[(df["status"] == 1) &
                     (df["frame_type"] == "dark") &
                     (df["vector_x"] != 0)].copy()    # vector_x != 0 ⟹ glint detectado
        if df_dark.empty:
            continue

        sess_name = os.path.splitext(os.path.basename(path))[0].replace("_segmentation_v2","")
        raw_dark_per_sess[sess_name] = df_dark.copy()

        t18 = df_dark[(df_dark["tar1_x"] == T18X) & (df_dark["tar1_y"] == T18Y)]
        if len(t18) >= MIN_TGT_FRAMES:
            t18_pupil_per_sess[sess_name] = (float(t18["pupil_x"].mean()),
                                              float(t18["pupil_y"].mean()))

    if not raw_dark_per_sess:
        return pd.DataFrame(), pd.DataFrame()

    # Referência global (média das T18 em todas as sessões)
    if t18_pupil_per_sess:
        ref_px = float(np.mean([v[0] for v in t18_pupil_per_sess.values()]))
        ref_py = float(np.mean([v[1] for v in t18_pupil_per_sess.values()]))
    else:
        ref_px = ref_py = 0.0

    # Passo 2 — aplica correção e agrega
    all_rows: List[dict] = []
    all_dark: List[pd.DataFrame] = []

    for sess_name, df_dark in raw_dark_per_sess.items():
        if sess_name in t18_pupil_per_sess:
            off_x = ref_px - t18_pupil_per_sess[sess_name][0]
            off_y = ref_py - t18_pupil_per_sess[sess_name][1]
        else:
            off_x = off_y = 0.0

        df_dark = df_dark.copy()
        # Posição corrigida = posição raw + offset de câmera
        df_dark["feat_x"] = df_dark["pupil_x"] + off_x
        df_dark["feat_y"] = df_dark["pupil_y"] + off_y
        df_dark["session"] = sess_name
        all_dark.append(df_dark)

        df_dark["_chg"] = (
            (df_dark["tar1_x"] != df_dark["tar1_x"].shift()) |
            (df_dark["tar1_y"] != df_dark["tar1_y"].shift())
        )
        df_dark["_seg"] = df_dark["_chg"].cumsum()

        for _, grp in df_dark.groupby("_seg", sort=False):
            tx = float(grp["tar1_x"].iloc[0])
            ty = float(grp["tar1_y"].iloc[0])
            if tx <= 0 or ty <= 0:
                continue
            tid = _tid(tx, ty)
            if tid == -1:
                continue
            # IQR filter on pupil position (not vector)
            seg = grp.copy()
            for col in ("feat_x","feat_y"):
                q1, q3 = seg[col].quantile(0.25), seg[col].quantile(0.75)
                iqr = q3 - q1
                seg = seg[(seg[col] >= q1-1.5*iqr) & (seg[col] <= q3+1.5*iqr)]
            if len(seg) < MIN_SEG_FRAMES:
                continue
            all_rows.append({
                "target_id": tid,
                "target_x":  tx,
                "target_y":  ty,
                "feat_x":    float(seg["feat_x"].mean()),
                "feat_y":    float(seg["feat_y"].mean()),
                # keep vector columns for compatibility
                "vector_x":  float(seg["vector_x"].mean()),
                "vector_y":  float(seg["vector_y"].mean()),
                "count":     len(seg),
            })

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.DataFrame(all_rows)
    df_calib = df_all.groupby("target_id").agg(
        feat_x=("feat_x",   "mean"),
        feat_y=("feat_y",   "mean"),
        vector_x=("vector_x","mean"),
        vector_y=("vector_y","mean"),
        target_x=("target_x","first"),
        target_y=("target_y","first"),
        count=("count","sum"),
    ).reset_index()
    df_calib = df_calib[df_calib["count"] >= MIN_TGT_FRAMES].reset_index(drop=True)

    df_frames = pd.concat(all_dark, ignore_index=True) if all_dark else pd.DataFrame()
    return df_calib, df_frames


def load_sessions_led_corrected(csv_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Como load_sessions_no_norm, mas aplica correção de deriva do LED entre sessões.

    O LED IR pode ser reposicionado entre sessões (fp1, fq1, …). Isso causa um
    offset sistemático no vetor P-CR por sessão que NÃO é movimento de cabeça.
    Correção: para cada sessão, computa a diferença entre o vetor médio de T18
    nessa sessão e o vetor médio global de T18.  Subtrai esse offset de TODOS os
    frames da sessão → vetores de todas as sessões ficam no mesmo referencial.

    Se uma sessão não tiver frames de T18 válidos, ela é usada sem correção
    (offset zero) — assumindo que o LED não foi movido nessa sessão.
    """
    T18X, T18Y = 960, 540

    # ── Passo 1: calcula offset T18 por sessão ───────────────────────────────
    t18_per_sess: Dict[str, Tuple[float, float]] = {}
    raw_dark_per_sess: Dict[str, pd.DataFrame] = {}

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"timestamp","status","vector_x","vector_y",
                "tar1_x","tar1_y","frame_type"}.issubset(df.columns):
            continue

        df_dark = df[(df["status"] == 1) &
                     (df["frame_type"] == "dark") &
                     (df["vector_x"] != 0)].copy()

        if {"pupil_x","pupil_y","glint_x","glint_y"}.issubset(df.columns):
            gd = np.sqrt((df_dark["pupil_x"] - df_dark["glint_x"])**2 +
                         (df_dark["pupil_y"] - df_dark["glint_y"])**2)
            df_dark = df_dark[gd <= MAX_GLINT_DIST]

        if df_dark.empty:
            continue

        sess_name = os.path.splitext(os.path.basename(path))[0].replace("_segmentation_v2","")
        raw_dark_per_sess[sess_name] = df_dark.copy()

        t18 = df_dark[(df_dark["tar1_x"] == T18X) & (df_dark["tar1_y"] == T18Y)]
        if len(t18) >= MIN_TGT_FRAMES:
            t18_per_sess[sess_name] = (float(t18["vector_x"].mean()),
                                       float(t18["vector_y"].mean()))

    if not raw_dark_per_sess:
        return pd.DataFrame(), pd.DataFrame()

    # Global T18 reference (mean across all sessions that have T18)
    if t18_per_sess:
        ref_x = float(np.mean([v[0] for v in t18_per_sess.values()]))
        ref_y = float(np.mean([v[1] for v in t18_per_sess.values()]))
    else:
        ref_x = ref_y = 0.0

    # ── Passo 2: aplica correção e agrega ────────────────────────────────────
    all_rows: List[dict] = []
    all_dark: List[pd.DataFrame] = []

    for sess_name, df_dark in raw_dark_per_sess.items():
        # Offset para esta sessão
        if sess_name in t18_per_sess:
            off_x = ref_x - t18_per_sess[sess_name][0]
            off_y = ref_y - t18_per_sess[sess_name][1]
        else:
            off_x = off_y = 0.0

        df_dark = df_dark.copy()
        df_dark["vector_x"] = df_dark["vector_x"] + off_x
        df_dark["vector_y"] = df_dark["vector_y"] + off_y
        df_dark["session"] = sess_name
        all_dark.append(df_dark)

        # Segmenta por alvo e calcula vetor médio por segmento
        df_dark["_chg"] = (
            (df_dark["tar1_x"] != df_dark["tar1_x"].shift()) |
            (df_dark["tar1_y"] != df_dark["tar1_y"].shift())
        )
        df_dark["_seg"] = df_dark["_chg"].cumsum()

        for _, grp in df_dark.groupby("_seg", sort=False):
            tx = float(grp["tar1_x"].iloc[0])
            ty = float(grp["tar1_y"].iloc[0])
            if tx <= 0 or ty <= 0:
                continue
            tid = _tid(tx, ty)
            if tid == -1:
                continue
            seg = _iqr_filter(grp)
            if len(seg) < MIN_SEG_FRAMES:
                continue
            all_rows.append({
                "target_id": tid,
                "target_x":  tx,
                "target_y":  ty,
                "vector_x":  float(seg["vector_x"].mean()),
                "vector_y":  float(seg["vector_y"].mean()),
                "count":     len(seg),
            })

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame()

    df_all = pd.DataFrame(all_rows)
    df_calib = df_all.groupby("target_id").agg(
        vector_x=("vector_x", "mean"),
        vector_y=("vector_y", "mean"),
        target_x=("target_x", "first"),
        target_y=("target_y", "first"),
        count=("count",    "sum"),
    ).reset_index()
    df_calib = df_calib[df_calib["count"] >= MIN_TGT_FRAMES].reset_index(drop=True)

    df_frames = pd.concat(all_dark, ignore_index=True) if all_dark else pd.DataFrame()
    return df_calib, df_frames


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — CALIBRAÇÃO POLINOMIAL + FILTRO EMA
# ═════════════════════════════════════════════════════════════════════════════

def calibrate(df_calib: pd.DataFrame) -> Optional[Tuple]:
    """
    Ajusta regressão Ridge polinomial grau 2 nos 9 alvos de treino.
    Retorna (df_results, model_x, model_y) ou None se dados insuficientes.
    """
    df_train = df_calib[df_calib["target_id"].isin(CALIB_IDS)]
    if len(df_train) < 4:
        return None

    X_train = df_train[["vector_x","vector_y"]].values
    X_all   = df_calib[["vector_x","vector_y"]].values

    mx = make_pipeline(PolynomialFeatures(POLY_DEGREE), Ridge(alpha=RIDGE_ALPHA))
    my = make_pipeline(PolynomialFeatures(POLY_DEGREE), Ridge(alpha=RIDGE_ALPHA))
    mx.fit(X_train, df_train["target_x"].values)
    my.fit(X_train, df_train["target_y"].values)

    df = df_calib.copy()
    df["pred_x"]   = mx.predict(X_all)
    df["pred_y"]   = my.predict(X_all)
    df["error_px"] = np.sqrt((df["pred_x"]-df["target_x"])**2 +
                              (df["pred_y"]-df["target_y"])**2)
    df["error_deg"] = np.degrees(
        np.arctan(df["error_px"] * PIXEL_PITCH / DISTANCE_MM)
    )
    df["split"] = df["target_id"].apply(
        lambda t: "train" if t in CALIB_IDS else "test"
    )
    return df, mx, my


def calibrate_tps(df_calib: pd.DataFrame) -> Optional[Tuple]:
    """
    Calibração via Thin-Plate Spline (TPS) nos 9 alvos de treino.

    Usa 'feat_x'/'feat_y' (posição corrigida da pupila) se disponíveis;
    caso contrário usa 'vector_x'/'vector_y' (vetor P-CR).

    TPS interpola exatamente nos pontos de treino (erro de treino ≈ 0).
    Retorna (df_results, interp_x, interp_y) ou None.
    """
    df_train = df_calib[df_calib["target_id"].isin(CALIB_IDS)]
    if len(df_train) < 4:
        return None

    # Usa posição de pupila corrigida se disponível (melhor sensibilidade)
    feat_cols = ["feat_x","feat_y"] if "feat_x" in df_calib.columns else ["vector_x","vector_y"]
    X_train = df_train[feat_cols].values
    X_all   = df_calib[feat_cols].values

    # RBFInterpolator com kernel thin_plate_spline e degree=1 (affine + TPS).
    # smoothing=0 → interpolação exata nos pontos de treino.
    interp_x = RBFInterpolator(X_train, df_train["target_x"].values,
                                kernel="thin_plate_spline", degree=1, smoothing=0)
    interp_y = RBFInterpolator(X_train, df_train["target_y"].values,
                                kernel="thin_plate_spline", degree=1, smoothing=0)

    df = df_calib.copy()
    df["pred_x"]    = interp_x(X_all)
    df["pred_y"]    = interp_y(X_all)
    df["error_px"]  = np.sqrt((df["pred_x"] - df["target_x"])**2 +
                               (df["pred_y"] - df["target_y"])**2)
    df["error_deg"] = np.degrees(
        np.arctan(df["error_px"] * PIXEL_PITCH / DISTANCE_MM)
    )
    df["split"] = df["target_id"].apply(
        lambda t: "train" if t in CALIB_IDS else "test"
    )
    return df, interp_x, interp_y


def apply_fixmean_to_frames(df_frames: pd.DataFrame,
                             interp_x, interp_y) -> pd.DataFrame:
    """
    Aplica TPS a cada dark frame e suaviza com a MÉDIA DA FIXAÇÃO.

    Para tarefas de fixação estática o filtro ótimo é a média dentro de cada
    segmento: elimina ruído frame-a-frame (σ_mean ≈ σ/√N) sem introduzir lag.
    Complementa com Savitzky-Golay (window=7, poly=2) para a tabela por frame.

    Colunas adicionadas:
        pred_x/y          — predição TPS por frame (bruta)
        smooth_x/y        — média da fixação (fixmean)
        sg_x/y            — Savitzky-Golay por frame
        error_deg_raw     — erro vs alvo, predição bruta
        error_deg_smooth  — erro vs alvo, fixmean
    """
    if df_frames.empty or interp_x is None:
        return df_frames

    df = df_frames.copy()
    feat_cols = ["feat_x","feat_y"] if "feat_x" in df.columns else ["vector_x","vector_y"]
    vecs = df[feat_cols].values
    df["pred_x"] = interp_x(vecs)
    df["pred_y"] = interp_y(vecs)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_chg"] = (
        (df["tar1_x"] != df["tar1_x"].shift()) |
        (df["tar1_y"] != df["tar1_y"].shift())
    )
    df["_seg"] = df["_chg"].cumsum()

    smooth_x = np.zeros(len(df))
    smooth_y = np.zeros(len(df))
    sg_x     = df["pred_x"].values.copy()
    sg_y     = df["pred_y"].values.copy()

    for _, grp in df.groupby("_seg", sort=False):
        idx = grp.index
        # Per-fixation mean
        smooth_x[idx] = grp["pred_x"].mean()
        smooth_y[idx] = grp["pred_y"].mean()
        # Savitzky-Golay (min window 5, polyorder 2)
        n = len(idx)
        if n >= 7:
            sg_x[idx] = savgol_filter(grp["pred_x"].values, window_length=7, polyorder=2)
            sg_y[idx] = savgol_filter(grp["pred_y"].values, window_length=7, polyorder=2)
        elif n >= 5:
            sg_x[idx] = savgol_filter(grp["pred_x"].values, window_length=5, polyorder=2)
            sg_y[idx] = savgol_filter(grp["pred_y"].values, window_length=5, polyorder=2)

    df["smooth_x"] = smooth_x
    df["smooth_y"] = smooth_y
    df["sg_x"]     = sg_x
    df["sg_y"]     = sg_y

    valid = (df["tar1_x"] > 0) & (df["tar1_y"] > 0)
    for px_col, err_col in [("pred_x",   "error_deg_raw"),
                             ("smooth_x", "error_deg_smooth")]:
        err = np.sqrt((df[px_col]   - df["tar1_x"])**2 +
                      (df["smooth_y" if "smooth" in px_col else "pred_y"] - df["tar1_y"])**2)
        df[err_col] = np.degrees(np.arctan(err * PIXEL_PITCH / DISTANCE_MM))
        df.loc[~valid, err_col] = np.nan
    return df


def apply_ema_to_frames(df_frames: pd.DataFrame,
                        mx, my,
                        alpha: float) -> pd.DataFrame:
    """
    Aplica modelo de calibração a cada dark frame e suaviza com EMA (α).

    Retorna df_frames enriquecido com:
        pred_x, pred_y        — gaze bruto por frame
        smooth_x, smooth_y    — gaze suavizado por EMA
        error_deg_raw         — erro angular bruto vs alvo
        error_deg_smooth      — erro angular suavizado vs alvo
    """
    if df_frames.empty or mx is None:
        return df_frames

    df = df_frames.copy()
    vecs = df[["vector_x","vector_y"]].values
    df["pred_x"] = mx.predict(vecs)
    df["pred_y"] = my.predict(vecs)

    # EMA dentro de cada segmento de alvo (reinicia em cada nova fixação)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_chg"] = (
        (df["tar1_x"] != df["tar1_x"].shift()) |
        (df["tar1_y"] != df["tar1_y"].shift())
    )
    df["_seg"] = df["_chg"].cumsum()

    smooth_x = np.zeros(len(df))
    smooth_y = np.zeros(len(df))

    for seg_id, grp in df.groupby("_seg", sort=False):
        idx = grp.index
        px  = grp["pred_x"].values
        py  = grp["pred_y"].values
        sx  = np.zeros(len(px)); sx[0] = px[0]
        sy  = np.zeros(len(py)); sy[0] = py[0]
        for i in range(1, len(px)):
            sx[i] = alpha * px[i] + (1-alpha) * sx[i-1]
            sy[i] = alpha * py[i] + (1-alpha) * sy[i-1]
        smooth_x[idx] = sx
        smooth_y[idx] = sy

    df["smooth_x"] = smooth_x
    df["smooth_y"] = smooth_y

    # Erro angular frame a frame (apenas frames com alvo válido)
    valid = (df["tar1_x"] > 0) & (df["tar1_y"] > 0)
    for col_p, col_s, col_raw, col_sm in [
        ("pred_x", "smooth_x", "error_deg_raw",    "error_deg_smooth"),
        ("pred_y", "smooth_y", "error_y_deg_raw",  "error_y_deg_smooth"),
    ]:
        pass  # calculamos abaixo em 2D

    err_raw = np.sqrt(
        (df["pred_x"]   - df["tar1_x"])**2 +
        (df["pred_y"]   - df["tar1_y"])**2
    )
    err_sm  = np.sqrt(
        (df["smooth_x"] - df["tar1_x"])**2 +
        (df["smooth_y"] - df["tar1_y"])**2
    )
    df["error_deg_raw"]    = np.degrees(np.arctan(err_raw * PIXEL_PITCH / DISTANCE_MM))
    df["error_deg_smooth"] = np.degrees(np.arctan(err_sm  * PIXEL_PITCH / DISTANCE_MM))

    # Zera erro em frames sem alvo definido
    df.loc[~valid, ["error_deg_raw","error_deg_smooth"]] = np.nan
    return df


def find_best_alpha(df_frames: pd.DataFrame, mx, my) -> Tuple[float, float]:
    """
    Busca o α EMA que minimiza o erro médio angular suavizado nos 26 alvos
    de teste. Retorna (best_alpha, best_error_deg).
    """
    best_alpha = EMA_ALPHAS[-1]
    best_err   = float("inf")

    for a in EMA_ALPHAS:
        dfe = apply_ema_to_frames(df_frames, mx, my, a)
        if dfe.empty or "error_deg_smooth" not in dfe.columns:
            continue
        valid = dfe[(dfe["tar1_x"] > 0) & (dfe["tar1_y"] > 0)].copy()
        if valid.empty:
            continue
        # Média do erro suavizado por alvo (último valor do segmento EMA)
        valid["_chg"] = (
            (valid["tar1_x"] != valid["tar1_x"].shift()) |
            (valid["tar1_y"] != valid["tar1_y"].shift())
        )
        valid["_seg"] = valid["_chg"].cumsum()
        seg_errs = []
        for _, grp in valid.groupby("_seg", sort=False):
            tid = _tid(float(grp["tar1_x"].iloc[0]), float(grp["tar1_y"].iloc[0]))
            if tid == -1 or tid in CALIB_IDS:
                continue   # só test targets para escolha de α
            seg_errs.append(float(grp["error_deg_smooth"].iloc[-1]))
        if not seg_errs:
            continue
        mean_e = float(np.mean(seg_errs))
        if mean_e < best_err:
            best_err   = mean_e
            best_alpha = a

    return best_alpha, best_err


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — EXPORTAÇÃO EXCEL
# ═════════════════════════════════════════════════════════════════════════════

def _excel_writer(path: str):
    """Cria ExcelWriter com openpyxl ou xlsxwriter."""
    try:
        import openpyxl  # noqa
        return pd.ExcelWriter(path, engine="openpyxl")
    except ImportError:
        return pd.ExcelWriter(path, engine="xlsxwriter")


def export_excel_segmentation(df_frames: pd.DataFrame,
                               subj: str, out_path: str) -> None:
    """
    Tabela 1 — Detecção (segmentação):
    Pupila, glint(s), vetor pupila-glint, métricas de qualidade.
    Apenas dark frames com status=1.
    """
    if df_frames.empty:
        return

    # Seleciona e renomeia colunas disponíveis
    col_map = {
        "session":       "Sessão",
        "file_name":     "Frame",
        "timestamp":     "Timestamp (s)",
        "frame_type":    "Tipo Frame",
        "tar1_x":        "Alvo X (px)",
        "tar1_y":        "Alvo Y (px)",
        "pupil_x":       "Pupila X (px)",
        "pupil_y":       "Pupila Y (px)",
        "pupil_radius":  "Raio Pupila (px)",
        "aspect_ratio":  "Razão Aspecto",
        "glint1_x":      "Glint1 X (px)",
        "glint1_y":      "Glint1 Y (px)",
        "glint2_x":      "Glint2 X (px)",
        "glint2_y":      "Glint2 Y (px)",
        "glint_x":       "Glint Médio X (px)",
        "glint_y":       "Glint Médio Y (px)",
        "vector_x":      "Vetor X (px)",
        "vector_y":      "Vetor Y (px)",
        "quality":       "Qualidade",
        "edge_support":  "Sup. Borda",
        "fit_rms":       "RMS Ajuste (px)",
        "n_inliers":     "N Inliers",
    }
    avail = {k: v for k, v in col_map.items() if k in df_frames.columns}
    df_out = df_frames[list(avail.keys())].rename(columns=avail)

    # Adiciona coluna Target ID
    if "Alvo X (px)" in df_out.columns and "Alvo Y (px)" in df_out.columns:
        df_out.insert(3, "Alvo ID",
            df_frames.apply(lambda r: _tid(r["tar1_x"], r["tar1_y"])
                            if r["tar1_x"] > 0 else -1, axis=1).values)

    with _excel_writer(out_path) as wr:
        df_out.to_excel(wr, sheet_name="Segmentação", index=False)
        try:
            ws = wr.sheets["Segmentação"]
            ws.freeze_panes = "A2"
        except Exception:
            pass
    print(f"        tabela_segmentacao.xlsx  ✓  ({len(df_out)} linhas)")


def export_excel_gaze(df_frames_ema: pd.DataFrame,
                      df_calib_result: pd.DataFrame,
                      subj: str, alpha: float, out_path: str) -> None:
    """
    Tabela 2 — Estimação de Gaze:
    Gaze bruto + suavizado por EMA + erro angular.
    """
    if df_frames_ema.empty:
        return

    cols_needed = ["session","file_name","timestamp","tar1_x","tar1_y",
                   "vector_x","vector_y","pred_x","pred_y",
                   "smooth_x","smooth_y","error_deg_raw","error_deg_smooth"]
    avail_cols = [c for c in cols_needed if c in df_frames_ema.columns]
    df_out = df_frames_ema[avail_cols].copy()

    rename = {
        "session":           "Sessão",
        "file_name":         "Frame",
        "timestamp":         "Timestamp (s)",
        "tar1_x":            "Alvo X (px)",
        "tar1_y":            "Alvo Y (px)",
        "vector_x":          "Vetor X (px)",
        "vector_y":          "Vetor Y (px)",
        "pred_x":            "Gaze Bruto X (px)",
        "pred_y":            "Gaze Bruto Y (px)",
        "smooth_x":          f"Gaze EMA X α={alpha:.2f} (px)",
        "smooth_y":          f"Gaze EMA Y α={alpha:.2f} (px)",
        "error_deg_raw":     "Erro Angular Bruto (°)",
        "error_deg_smooth":  "Erro Angular EMA (°)",
    }
    df_out.rename(columns=rename, inplace=True)

    # Adiciona ID do alvo
    if "Alvo X (px)" in df_out.columns:
        df_out.insert(3, "Alvo ID",
            df_frames_ema.apply(lambda r: _tid(r["tar1_x"], r["tar1_y"])
                                if r["tar1_x"] > 0 else -1, axis=1).values)

    with _excel_writer(out_path) as wr:
        df_out.to_excel(wr, sheet_name="Estimação Gaze", index=False)
        # Aba de resumo de calibração
        if df_calib_result is not None and not df_calib_result.empty:
            sum_cols = ["target_id","target_x","target_y",
                        "pred_x","pred_y","error_px","error_deg","split"]
            avail_s = [c for c in sum_cols if c in df_calib_result.columns]
            df_calib_result[avail_s].rename(columns={
                "target_id": "Alvo ID", "target_x": "Alvo X", "target_y": "Alvo Y",
                "pred_x": "Previsto X", "pred_y": "Previsto Y",
                "error_px": "Erro (px)", "error_deg": "Erro (°)", "split": "Split",
            }).to_excel(wr, sheet_name="Calibração", index=False)
        try:
            wr.sheets["Estimação Gaze"].freeze_panes = "A2"
        except Exception:
            pass
    print(f"        tabela_estimacao.xlsx     ✓  ({len(df_out)} linhas)")


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — VISUALIZAÇÕES
# ═════════════════════════════════════════════════════════════════════════════

def save_heatmap(df_results: pd.DataFrame, title: str, path: str) -> None:
    heat   = np.full((5, 7), np.nan)
    annot  = np.empty((5, 7), dtype=object)
    for _, row in df_results.iterrows():
        tid = int(row["target_id"])
        r, c = (tid-1)//7, (tid-1)%7
        heat[r, c]  = row["error_deg"]
        annot[r, c] = f"{row['error_deg']:.2f}°"
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#555555")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(np.ma.masked_invalid(heat), annot=annot, fmt="", cmap=cmap,
                vmin=0, vmax=4.0, linewidths=1.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Erro Angular (°)"},
                annot_kws={"fontsize": 10, "color": "white", "weight": "bold"})
    for tid in CALIB_IDS:
        r, c = (tid-1)//7, (tid-1)%7
        ax.add_patch(plt.Rectangle((c,r),1,1,fill=False,edgecolor="red",lw=2.5))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Coluna"); ax.set_ylabel("Linha")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_aggregate_heatmap(all_results: List[dict], out_dir: str) -> None:
    if len(all_results) < 2:
        return
    target_errs: Dict[int, List[float]] = {}
    for r in all_results:
        for _, row in r["df"].iterrows():
            target_errs.setdefault(int(row["target_id"]), []).append(
                float(row["error_deg"]))

    rows_agg = [{"target_id": tid,
                 "error_deg": float(np.mean(v)) if v else np.nan,
                 "std_deg":   float(np.std(v))  if v else np.nan}
                for tid, v in [(t, target_errs.get(t,[])) for t in range(1,36)]]
    df_agg = pd.DataFrame(rows_agg)

    heat  = np.full((5,7), np.nan)
    annot = np.empty((5,7), dtype=object)
    for _, row in df_agg.iterrows():
        tid = int(row["target_id"])
        ri, ci = (tid-1)//7, (tid-1)%7
        if not np.isnan(row["error_deg"]):
            heat[ri,ci]  = row["error_deg"]
            annot[ri,ci] = f"{row['error_deg']:.2f}°\n±{row['std_deg']:.2f}"
        else:
            annot[ri,ci] = "—"

    n = len(all_results)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#555555")
    fig, ax = plt.subplots(figsize=(15,8))
    sns.heatmap(np.ma.masked_invalid(heat), annot=annot, fmt="",
                cmap=cmap, vmin=0, vmax=4.0, linewidths=1.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Erro médio (°)"},
                annot_kws={"fontsize":9, "color":"white", "weight":"bold"})
    for tid in CALIB_IDS:
        r,c=(tid-1)//7,(tid-1)%7
        ax.add_patch(plt.Rectangle((c,r),1,1,fill=False,edgecolor="red",lw=2.5))
    ax.set_title(f"Erro Angular Agregado — {n} Sujeitos (média ± DP)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Coluna"); ax.set_ylabel("Linha")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"heatmap_agregado_15.png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  heatmap_agregado_15.png  ✓")


# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 6 — RELATÓRIO FINAL
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: List[dict]) -> None:
    if not all_results:
        print("\n[AVISO] Nenhum sujeito processado.")
        return
    print(f"\n{'='*80}")
    print(" TABELA RESUMO — 15 USUÁRIOS (SESSÃO 1, SEM NORMALIZAÇÃO T18)")
    print(f"{'='*80}")
    print(f" {'#':>3} {'Sujeito':<14} {'TPS (mean)':>11} {'Fixmean':>9} "
          f"{'Train':>7} {'Test':>7} {'Alvos':>6} {'Area(px²)':>10}")
    print(f" {'─'*3} {'─'*14} {'─'*11} {'─'*9} {'─'*7} {'─'*7} {'─'*6} {'─'*10}")
    for i, r in enumerate(all_results, 1):
        fm = r.get("ema_deg", float("nan"))
        ok = "✓" if fm < 1.5 else ("~" if fm < 3.0 else " ")
        area = r.get("mean_area", float("nan"))
        print(f"{ok}{i:>3} {r['subject']:<14} "
              f"{r['mean_deg']:>10.3f}° "
              f"{fm:>8.3f}° "
              f"{r['train_deg']:>6.3f}° "
              f"{r['test_deg']:>6.3f}° "
              f"{r['n_targets']:>6} "
              f"{area:>10.0f}")

    gm = float(np.mean([r["mean_deg"]  for r in all_results]))
    ge = float(np.mean([r.get("ema_deg",r["mean_deg"]) for r in all_results]))
    gs = float(np.std( [r["mean_deg"]  for r in all_results]))
    n15 = sum(1 for r in all_results if r.get("ema_deg",99) < 1.5)
    n20 = sum(1 for r in all_results if r.get("ema_deg",99) < 2.0)
    print(f" {'─'*3} {'─'*14} {'─'*11} {'─'*9} {'─'*7} {'─'*7} {'─'*6} {'─'*10}")
    print(f"    {'GLOBAL':<14} {gm:>10.3f}° {ge:>8.3f}°")
    print(f"\n  Sujeitos com Fixmean < 1.5°: {n15}/{len(all_results)}")
    print(f"  Sujeitos com Fixmean < 2.0°: {n20}/{len(all_results)}")
    print(f"{'='*80}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline 15 usuários — sessão 1, chin rest, sem T18 norm"
    )
    parser.add_argument("--only", nargs="*", metavar="USER",
                        help="Processa apenas estes sujeitos")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Reprocessa segmentação mesmo com cache")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_global = time.time()

    users = args.only if args.only else USERS

    all_results: List[dict] = []

    for subj in users:
        if subj not in USERS:
            print(f"\n[WARN] {subj} não está na lista de 15 usuários — ignorado.")
            continue

        print(f"\n{'─'*70}")
        print(f"  Sujeito: {subj}  (#{USERS.index(subj)+1}/15)")
        subj_out = os.path.join(args.output_dir, subj)
        os.makedirs(subj_out, exist_ok=True)

        # ── 1. Segmentação ────────────────────────────────────────────────
        print("  [1/4] Segmentação:")
        seg_csvs: List[str] = []
        for sess in SESSION1:
            sess_dir = find_session_dir(args.data_dir, subj, sess)
            if sess_dir is None:
                print(f"      [SKIP] {sess}: sessão não encontrada")
                continue
            out_csv = os.path.join(subj_out, f"{sess}_segmentation_v2.csv")
            csv_path = run_segmentation(sess_dir, out_csv, force=args.force)
            if csv_path:
                seg_csvs.append(csv_path)

        if not seg_csvs:
            print("  [SKIP] Sem CSVs de segmentação.")
            continue

        # ── 2. Carrega dados (vetor P-CR sem normalização, chin rest) ────
        print("  [2/4] Carregando dados (P-CR vector, sem normalização):")
        df_calib, df_frames = load_sessions_no_norm(seg_csvs)

        if df_calib.empty:
            print("        Sem detecções válidas.")
            continue

        n_tgt = len(df_calib)
        print(f"        {n_tgt}/35 alvos com dados  "
              f"({len(df_frames)} frames dark válidos)")

        # ── 3. Calibração TPS + filtro de média por fixação ────────────
        print("  [3/4] Calibração TPS + fixmean:")
        result = calibrate_tps(df_calib)
        if result is None:
            print(f"        Alvos de treino insuficientes.")
            continue

        df_cal, interp_x, interp_y = result
        mean_e  = float(df_cal["error_deg"].mean())
        std_e   = float(df_cal["error_deg"].std())
        train_e = float(df_cal[df_cal["split"]=="train"]["error_deg"].mean())
        test_e  = float(df_cal[df_cal["split"]=="test"]["error_deg"].mean()
                        if (df_cal["split"]=="test").any() else np.nan)
        print(f"        TPS:   {mean_e:.3f}° ± {std_e:.3f}°   "
              f"train={train_e:.3f}°  test={test_e:.3f}°"
              f"  {'✓ <1.5°' if mean_e < 1.5 else ('✓ <3°' if mean_e < 3.0 else '')}")

        # Aplica TPS + fixmean a cada frame
        df_frames_sm = apply_fixmean_to_frames(df_frames, interp_x, interp_y)

        # Erro médio suavizado (fixmean) para o relatório
        if not df_frames_sm.empty and "error_deg_smooth" in df_frames_sm.columns:
            sm_e = float(df_frames_sm[
                (df_frames_sm["tar1_x"]>0) & (df_frames_sm["tar1_y"]>0)
            ]["error_deg_smooth"].mean())
        else:
            sm_e = mean_e
        print(f"        Fixmean: {sm_e:.3f}°"
              f"  {'✓ <1.5°' if sm_e < 1.5 else ('✓ <2°' if sm_e < 2.0 else '✗')}")

        # Área média da pupila (px²)
        if "pupil_radius" in df_frames.columns:
            mean_area = float((np.pi * df_frames["pupil_radius"]**2).mean())
        else:
            mean_area = float("nan")

        # ── 4. Salva resultados ────────────────────────────────────────
        print("  [4/4] Salvando:")
        df_cal.to_csv(os.path.join(subj_out, "calibration_results.csv"), index=False)

        save_heatmap(
            df_cal,
            f"{subj}  —  TPS: {mean_e:.2f}°  Fixmean: {sm_e:.2f}°",
            os.path.join(subj_out, "heatmap.png"),
        )
        print(f"        calibration_results.csv  ✓")
        print(f"        heatmap.png               ✓")

        export_excel_segmentation(
            df_frames,
            subj,
            os.path.join(subj_out, "tabela_segmentacao.xlsx"),
        )
        export_excel_gaze(
            df_frames_sm,
            df_cal,
            subj,
            0.0,   # alpha não se aplica ao fixmean
            os.path.join(subj_out, "tabela_estimacao.xlsx"),
        )

        all_results.append({
            "subject":    subj,
            "mean_deg":   mean_e,
            "std_deg":    std_e,
            "train_deg":  train_e,
            "test_deg":   test_e,
            "n_targets":  n_tgt,
            "ema_deg":    sm_e,     # fixmean error (used in summary as "filtered")
            "alpha":      0.0,
            "mean_area":  mean_area,
            "df":         df_cal,
        })

    # ── Relatório Final ────────────────────────────────────────────────────
    print_summary(all_results)

    if all_results:
        summary = [{
            "sujeito":              r["subject"],
            "erro_tps_deg":         round(r["mean_deg"],  3),
            "desvio_padrao_deg":    round(r["std_deg"],   3),
            "erro_treino_deg":      round(r["train_deg"], 3),
            "erro_teste_deg":       round(r["test_deg"],  3),
            "erro_fixmean_deg":     round(r["ema_deg"],   3),
            "area_pupila_px2":      round(r.get("mean_area", float("nan")), 1),
            "n_alvos":              r["n_targets"],
        } for r in all_results]

        gm = float(np.mean([r["mean_deg"]  for r in all_results]))
        ge = float(np.mean([r["ema_deg"]   for r in all_results]))
        gs = float(np.std( [r["mean_deg"]  for r in all_results]))
        summary.append({
            "sujeito": "GLOBAL",
            "erro_tps_deg": round(gm, 3),
            "desvio_padrao_deg": round(gs, 3),
            "erro_fixmean_deg": round(ge, 3),
        })
        pd.DataFrame(summary).to_csv(
            os.path.join(args.output_dir, "summary_15usuarios.csv"), index=False
        )
        print(f"\n  summary_15usuarios.csv  ✓")
        save_aggregate_heatmap(all_results, args.output_dir)

    elapsed = (time.time() - t_global) / 60
    print(f"\n[Pipeline] Concluído em {elapsed:.1f} min")


if __name__ == "__main__":
    main()
