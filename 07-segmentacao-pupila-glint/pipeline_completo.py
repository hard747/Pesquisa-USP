#!/usr/bin/env python3
"""
Pipeline Completo — Segmentação v2 + Calibração + Estimação
=============================================================
Executa de ponta a ponta, por sujeito:

  Etapa 1 — Segmentação (pupil_glint_detector_v2.py)
    - Detecta pupila + glints em cada sessão de fixação (fp*/fq*/…/fv*)
    - Gera CSV com vetor pupila-glint fundido com dados de alvo
    - Cache: pula sessões já processadas

  Etapa 2 — Calibração (Polynomial Ridge Regression)
    - Agrupa frames por alvo, remove bordas temporais e outliers IQR
    - Treina mapeamento (vector_x, vector_y) → (tela_x, tela_y)
      usando 9 alvos de calibração predefinidos (grade 3×3)
    - Avalia nos 26 alvos restantes (held-out)

  Etapa 3 — Relatório
    - Erro angular médio ± desvio padrão por sujeito
    - Separação treino/teste
    - Mapa de calor Viridis + scatter diagram
    - Tabela CSV final agregada

Uso:
    python3 pipeline_completo.py                        # todos os sujeitos
    python3 pipeline_completo.py --subjects P16 P17  # seleção
    python3 pipeline_completo.py --force                # reprocessa mesmo com cache
"""

import sys, os, math, argparse, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # headless — sem janela
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ── importa pipeline de segmentação v2 ───────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from pupil_glint_detector_v2 import (
    EyeTrackingPipeline, export_csv, FrameResult,
    NATIVE_W, NATIVE_H,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO GERAL
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "data"
OUTPUT_DIR = "results_pipeline_v2"

# Geometria do experimento
SCREEN_WIDTH_PX  = 1920
SCREEN_HEIGHT_PX = 1080
SCREEN_WIDTH_MM  = 507.0          # largura física do monitor em mm
DISTANCE_MM      = 600.0          # distância sujeito-tela em mm
PIXEL_PITCH_MM   = SCREEN_WIDTH_MM / SCREEN_WIDTH_PX   # 0.2641 mm/px
# 1° de ângulo visual ≈ 39.7 px neste setup
ONE_DEG_PX       = DISTANCE_MM * math.tan(math.radians(1)) / PIXEL_PITCH_MM

# Grade de alvos: 5 linhas × 7 colunas = 35 alvos
COLS = [135, 410, 685, 960, 1235, 1510, 1785]
ROWS = [108, 324, 540, 756, 972]

# 9 alvos usados para treinar o modelo (grade 3×3 dispersa).
# Os 26 restantes são avaliação pura (held-out).
CALIBRATION_TARGET_IDS = {1, 4, 7, 15, 18, 21, 29, 32, 35}

# Sessões de fixação: fp*, fq*, fr*, fs*, ft*, fu*, fv*
FIXATION_PREFIXES = ("fp", "fq", "fr", "fs", "ft", "fu", "fv")
VALID_IMG_EXT     = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".pgm")

# Parâmetros do modelo
POLY_DEGREE  = 2
RIDGE_ALPHA  = 5.0
MIN_FRAMES_PER_TARGET = 3   # descartar alvo com menos frames que isso
MIN_CALIB_TARGETS     = 5   # mínimo de alvos de treino para aceitar sujeito


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — DESCOBERTA DE SUJEITOS E SESSÕES
# ─────────────────────────────────────────────────────────────────────────────

def discover_subjects(data_dir: str,
                      filter_list: Optional[List[str]] = None
                      ) -> Dict[str, List[str]]:
    """
    Varre data_dir e retorna {sujeito: [session_dir, ...]} para cada
    sujeito que tenha ao menos uma sessão de fixação com imagens + CSV de coleta.
    """
    subjects: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, name)
        if not os.path.isdir(subj_dir):
            continue
        if filter_list and name not in filter_list:
            continue
        sessions = []
        for sess in sorted(os.listdir(subj_dir)):
            if not any(sess.startswith(p) for p in FIXATION_PREFIXES):
                continue
            sess_dir = os.path.join(subj_dir, sess)
            if not os.path.isdir(sess_dir):
                continue
            imgs = [f for f in os.listdir(sess_dir)
                    if f.lower().endswith(VALID_IMG_EXT)]
            csvs = [f for f in os.listdir(sess_dir)
                    if f.lower().endswith(".csv")
                    and "processed" not in f.lower()
                    and "segmentation" not in f.lower()]
            if imgs and csvs:
                sessions.append(sess_dir)
        if sessions:
            subjects[name] = sessions
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — SEGMENTAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def run_segmentation(session_dir: str,
                     out_dir: str,
                     force: bool = False) -> Optional[str]:
    """
    Roda detecção v2 em uma sessão e retorna o caminho do CSV resultante.
    Se o CSV já existir e force=False, devolve o caminho sem re-processar.
    """
    sess_name = os.path.basename(session_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{sess_name}_segmentation_v2.csv")

    if os.path.exists(out_csv) and not force:
        # Conta válidos para print informativo
        try:
            df = pd.read_csv(out_csv)
            n_valid = int((df["status"] == 1).sum())
            n_total = len(df)
            print(f"    [cache] {sess_name}: {n_valid}/{n_total} válidos "
                  f"({n_valid/n_total:.1%})")
        except Exception:
            print(f"    [cache] {sess_name}")
        return out_csv

    # Descobre imagens e CSV de entrada
    files = sorted(f for f in os.listdir(session_dir)
                   if f.lower().endswith(VALID_IMG_EXT))
    if not files:
        print(f"    [SKIP]  {sess_name}: sem imagens")
        return None

    coleta_csvs = [f for f in os.listdir(session_dir)
                   if f.lower().endswith(".csv")
                   and "processed" not in f.lower()
                   and "segmentation" not in f.lower()]
    input_csv = os.path.join(session_dir, coleta_csvs[0]) if coleta_csvs else None

    # Roda o pipeline frame a frame
    pipeline    = EyeTrackingPipeline()
    results: List[FrameResult] = []
    t0 = time.time()

    for idx, filename in enumerate(files):
        bgr = cv2.imread(os.path.join(session_dir, filename))
        if bgr is None:
            continue
        if bgr.shape[:2] != (NATIVE_H, NATIVE_W):
            bgr = cv2.resize(bgr, (NATIVE_W, NATIVE_H))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        results.append(pipeline.process(gray, filename, idx + 1))

    if not results:
        print(f"    [FAIL]  {sess_name}: nenhum frame processado")
        return None

    export_csv(results, out_csv, input_csv)

    st  = pipeline.stats()
    elapsed = time.time() - t0
    fps = len(files) / elapsed if elapsed > 0 else 0
    print(f"    {sess_name}: {st['valid']}/{st['total']} válidos "
          f"({st['valid_rate']:.1%})   {fps:.0f} fps")
    return out_csv


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — CALIBRAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_id(x: float, y: float) -> int:
    """Converte (x, y) de tela em ID de alvo (1–35) ou -1 se não bater."""
    try:
        c = COLS.index(int(round(x)))
        r = ROWS.index(int(round(y)))
        return r * 7 + c + 1
    except ValueError:
        return -1


def _process_segment(seg: pd.DataFrame) -> Optional[dict]:
    """
    Corte temporal + remoção IQR de outliers → média do vetor.
    Retorna dict com vector_x/y e count, ou None se dados insuficientes.
    """
    if len(seg) < 6:
        return None
    t0, t1 = float(seg["timestamp"].iloc[0]), float(seg["timestamp"].iloc[-1])
    dur   = t1 - t0
    trim  = 0.5 if dur > 1.5 else 0.2
    seg   = seg[(seg["timestamp"] >= t0 + trim) & (seg["timestamp"] <= t1 - trim)]
    if seg.empty:
        return None
    seg = seg[seg["status"] == 1]
    if seg.empty:
        return None
    for col in ("vector_x", "vector_y"):
        q1, q3 = seg[col].quantile(0.25), seg[col].quantile(0.75)
        iqr = q3 - q1
        seg = seg[(seg[col] >= q1 - 1.5 * iqr) & (seg[col] <= q3 + 1.5 * iqr)]
    if len(seg) < MIN_FRAMES_PER_TARGET:
        return None
    return {
        "vector_x": float(seg["vector_x"].mean()),
        "vector_y": float(seg["vector_y"].mean()),
        "count":    len(seg),
    }


T18_X, T18_Y = 960, 540          # centro da tela — âncora de normalização
MIN_T18_FRAMES = 5               # mínimo de frames de T18 para normalizar sessão
MIN_PASS_FRAMES_PER_SESSION = 0  # 0 = desabilitado; definir > 0 para excluir sessões
                                 # com falha do classificador de frames (ex: ft2 P16
                                 # tem apenas 5 dark frames devido a Otsu incorreto).
                                 # Valor 80 remove esses casos mas reduz cobertura.
MAX_GLINT_DIST = 60              # px — rejeita glints visivelmente errados
                                 # (reflexo secundário LED ou inclinação de cabeça)
MIN_TOTAL_FRAMES_PER_TARGET = 5   # descarta alvo com menos frames TOTAIS que isso


def load_and_aggregate(csv_paths: List[str]) -> pd.DataFrame:
    """
    Lê todos os CSVs de sessão de um sujeito, segmenta por alvo,
    remove bordas e outliers, e retorna uma linha por alvo com vetor médio.

    Normalização por T18 (compensação de deriva entre sessões)
    -----------------------------------------------------------
    Sem fixação de cabeça, o vetor pupila-glint muda entre sessões porque
    o sujeito se reposiciona. O alvo T18 (centro, 960×540) aparece em TODAS
    as 14 sessões de fixação. Subtraindo o vetor médio de T18 de cada sessão
    alinhamos todas as sessões a um referencial comum independente da posição
    da cabeça (calibração relativa, padrão em P-CR).

    Filtros adicionais
    ------------------
    - vector_x/y == 0 → frame sem glint detectado (status=1 mas sem reflexo)
    - distância glint-pupila > MAX_GLINT_DIST → detecção provavelmente errada
    """
    all_rows = []

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        needed = {"timestamp", "status", "vector_x", "vector_y",
                  "tar1_x", "tar1_y", "pupil_x", "pupil_y",
                  "glint_x", "glint_y"}
        if not needed.issubset(df.columns):
            continue

        # ── Filtros de qualidade ─────────────────────────────────────
        # 1) Só frames com pupila E glint válidos, apenas DARK frames.
        #
        #    No sistema estroboscópico:
        #    - Dark frame: LED IR desligado → pupila escura, glint fraco/ambiente
        #      → glint consistente entre sessões (fonte de luz ambiente estável)
        #    - Bright frame: LED IR ligado → múltiplos reflexos fortes (óculos,
        #      lentes, superfícies) → glint instável; variação de 65 px entre sessões
        #
        #    Usar SOMENTE dark frames elimina essa fonte de ruído e torna o
        #    vetor pupila-glint aproximadamente estável para a mesma direção de olhar.
        df = df[(df["status"] == 1) &
                (df["frame_type"] == "dark") &
                (df["vector_x"] != 0)].copy()

        # 2) Rejeita glints obviamente errados (> MAX_GLINT_DIST do centro da pupila)
        glint_dist = np.sqrt((df["pupil_x"] - df["glint_x"]) ** 2 +
                             (df["pupil_y"] - df["glint_y"]) ** 2)
        df = df[glint_dist <= MAX_GLINT_DIST]

        if df.empty:
            continue

        # 3) Filtro de qualidade da sessão (opcional).
        #    Sessões com poucos frames indicam falha do classificador de frames
        #    (FrameClassifier coloca quase tudo como "bright") ou glint instável.
        #    Aumentar MIN_PASS_FRAMES_PER_SESSION para excluir sessões problemáticas
        #    melhora P16 mas reduz cobertura de alvos para outros sujeitos.
        if MIN_PASS_FRAMES_PER_SESSION > 0 and len(df) < MIN_PASS_FRAMES_PER_SESSION:
            continue

        # ── Normalização por T18 (deriva entre sessões) ──────────────
        t18_frames = df[(df["tar1_x"] == T18_X) & (df["tar1_y"] == T18_Y)]
        if len(t18_frames) < MIN_T18_FRAMES:
            # Sessão sem dados suficientes do alvo central → pula
            # (sem T18 não conseguimos compensar a deriva de cabeça)
            continue
        t18_vx = float(t18_frames["vector_x"].mean())
        t18_vy = float(t18_frames["vector_y"].mean())

        # Vetor normalizado = vetor bruto − âncora T18
        # Depois da normalização, T18 mapeia para vector≈(0, 0)
        df = df.copy()
        df["vector_x"] -= t18_vx
        df["vector_y"] -= t18_vy

        # ── Segmentação por alvo ─────────────────────────────────────
        df["_chg"] = ((df["tar1_x"] != df["tar1_x"].shift()) |
                      (df["tar1_y"] != df["tar1_y"].shift()))
        df["_seg"] = df["_chg"].cumsum()

        for _, grp in df.groupby("_seg", sort=False):
            tx = float(grp["tar1_x"].iloc[0])
            ty = float(grp["tar1_y"].iloc[0])
            if tx <= 0 or ty <= 0:
                continue
            tid = _get_target_id(tx, ty)
            if tid == -1:
                continue
            res = _process_segment(grp)
            if res:
                all_rows.append({"target_id": tid,
                                 "target_x":  tx,
                                 "target_y":  ty,
                                 **res})

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)
    # Média entre sessões para o mesmo alvo
    df_agg = df_all.groupby("target_id").agg(
        vector_x=("vector_x", "mean"),
        vector_y=("vector_y", "mean"),
        target_x=("target_x", "first"),
        target_y=("target_y", "first"),
        count=("count",    "sum"),
    ).reset_index()
    # Remove alvos com poucos frames acumulados (ruído puro)
    df_agg = df_agg[df_agg["count"] >= MIN_TOTAL_FRAMES_PER_TARGET].reset_index(drop=True)
    return df_agg


def calibrate(df_targets: pd.DataFrame,
              degree: int = POLY_DEGREE,
              alpha:  float = RIDGE_ALPHA,
              ) -> Optional[Tuple[pd.DataFrame, object, object]]:
    """
    Ajusta regressão polinomial de grau `degree` com regularização Ridge.
    Treina nos CALIBRATION_TARGET_IDS, avalia em todos os 35.
    Retorna (df_results, model_x, model_y) ou None se dados insuficientes.
    """
    df_train = df_targets[df_targets["target_id"].isin(CALIBRATION_TARGET_IDS)]
    if len(df_train) < MIN_CALIB_TARGETS:
        return None

    X_train = df_train[["vector_x", "vector_y"]].values
    X_all   = df_targets[["vector_x", "vector_y"]].values

    model_x = make_pipeline(PolynomialFeatures(degree, include_bias=True),
                             Ridge(alpha=alpha))
    model_y = make_pipeline(PolynomialFeatures(degree, include_bias=True),
                             Ridge(alpha=alpha))
    model_x.fit(X_train, df_train["target_x"].values)
    model_y.fit(X_train, df_train["target_y"].values)

    df = df_targets.copy()
    df["pred_x"]    = model_x.predict(X_all)
    df["pred_y"]    = model_y.predict(X_all)
    df["error_px"]  = np.sqrt((df["pred_x"] - df["target_x"])**2 +
                               (df["pred_y"] - df["target_y"])**2)
    # Erro angular: arctan(distância_mm / distância_mm_ao_olho)
    df["error_deg"] = np.degrees(
        np.arctan(df["error_px"] * PIXEL_PITCH_MM / DISTANCE_MM)
    )
    df["split"] = df["target_id"].apply(
        lambda t: "train" if t in CALIBRATION_TARGET_IDS else "test"
    )
    return df, model_x, model_y


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 4 — VISUALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap_data(df_results: pd.DataFrame
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Monta matrizes 5×7 de erro e anotações para o mapa de calor."""
    heat   = np.full((5, 7), np.nan)
    annot  = np.empty((5, 7), dtype=object)
    for _, row in df_results.iterrows():
        tid = int(row["target_id"])
        r, c = (tid - 1) // 7, (tid - 1) % 7
        heat[r, c] = row["error_deg"]
        annot[r, c] = f"{row['error_deg']:.2f}°"
    return heat, annot


def save_heatmap(df_results: pd.DataFrame, title: str, path: str) -> None:
    heat, annot = _heatmap_data(df_results)
    fig, ax = plt.subplots(figsize=(14, 7))
    # NaN → cinza (sem dados)
    masked = np.ma.masked_invalid(heat)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#555555")
    sns.heatmap(masked, annot=annot, fmt="", cmap=cmap,
                vmin=0, vmax=2.0,
                linewidths=1.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Erro (graus)"},
                annot_kws={"fontsize": 10, "color": "white", "weight": "bold"})
    # Marca alvos de calibração (borda vermelha)
    for tid in CALIBRATION_TARGET_IDS:
        r, c = (tid - 1) // 7, (tid - 1) % 7
        ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                   fill=False, edgecolor="red", lw=2.5))
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Coluna (x)", fontsize=12)
    ax.set_ylabel("Linha (y)", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scatter(df_results: pd.DataFrame, title: str, path: str) -> None:
    """
    Scatter dos alvos reais (●) e previstos (✕) por split (treino/teste).
    Linhas ligam cada par real→previsto.
    """
    colors = {"train": "#2196F3", "test": "#FF9800"}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    for _, row in df_results.iterrows():
        col = colors.get(row["split"], "#aaaaaa")
        ax.plot([row["target_x"], row["pred_x"]],
                [row["target_y"], row["pred_y"]],
                color=col, alpha=0.6, linewidth=0.8)
        ax.scatter(row["target_x"], row["target_y"],
                   color=col, s=40, zorder=3)
        ax.scatter(row["pred_x"],   row["pred_y"],
                   color=col, marker="x", s=55, linewidths=1.4, zorder=3)

    # Legenda
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["train"],
               markersize=8, label=f"Treino ({POLY_DEGREE}° grau)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["test"],
               markersize=8, label="Teste (held-out)"),
        Line2D([0], [0], marker="x", color="w", markersize=8,
               markeredgewidth=1.4, label="Previsto"),
    ]
    ax.legend(handles=handles, facecolor="#2a2a4a", edgecolor="#555",
              labelcolor="white", fontsize=9)
    ax.set_xlim(0, SCREEN_WIDTH_PX)
    ax.set_ylim(SCREEN_HEIGHT_PX, 0)
    ax.set_title(title, fontsize=13, color="white", fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 5 — RELATÓRIO FINAL
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: List[dict]) -> None:
    if not all_results:
        print("\n[AVISO] Nenhum sujeito processado com sucesso.")
        return

    print(f"\n{'='*72}")
    print(" TABELA RESUMO — ERRO ANGULAR POR SUJEITO")
    print(f"{'='*72}")
    print(f" {'Sujeito':<18} {'Erro Médio':>10} {'±DP':>7} "
          f"{'Treino':>8} {'Teste':>8} {'Alvos':>6}")
    print(f" {'-'*18} {'-'*10} {'-'*7} {'-'*8} {'-'*8} {'-'*6}")

    for r in sorted(all_results, key=lambda x: x["mean_deg"]):
        ok = "✓" if r["mean_deg"] < 1.0 else " "
        print(f"{ok} {r['subject']:<18} {r['mean_deg']:>9.3f}° "
              f"{r['std_deg']:>6.3f}° "
              f"{r['train_deg']:>7.3f}° "
              f"{r['test_deg']:>7.3f}° "
              f"{r['n_targets']:>6}")

    global_mean = float(np.mean([r["mean_deg"] for r in all_results]))
    global_std  = float(np.std( [r["mean_deg"] for r in all_results]))
    n_below_1   = sum(1 for r in all_results if r["mean_deg"] < 1.0)
    print(f" {'─'*18} {'─'*10} {'─'*7} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  {'GLOBAL':<18} {global_mean:>9.3f}° {global_std:>6.3f}°")
    print(f"\n  Sujeitos < 1°: {n_below_1}/{len(all_results)}")
    print(f"{'='*72}")


def save_aggregate_heatmap(all_results: List[dict], out_dir: str) -> None:
    """Mapa de calor da média (± DP) de erro por alvo, todos os sujeitos."""
    if len(all_results) < 2:
        return
    target_errors: Dict[int, List[float]] = {}
    for r in all_results:
        for _, row in r["df"].iterrows():
            tid = int(row["target_id"])
            target_errors.setdefault(tid, []).append(float(row["error_deg"]))

    rows_agg = []
    for tid in range(1, 36):
        errs = target_errors.get(tid, [])
        rows_agg.append({
            "target_id": tid,
            "error_deg": float(np.mean(errs)) if errs else np.nan,
            "std_deg":   float(np.std(errs))  if errs else np.nan,
        })
    df_agg = pd.DataFrame(rows_agg)

    # Heatmap anotado com "média ± DP"
    heat  = np.full((5, 7), np.nan)
    annot = np.empty((5, 7), dtype=object)
    for _, row in df_agg.iterrows():
        tid = int(row["target_id"])
        r_i, c_i = (tid - 1) // 7, (tid - 1) % 7
        if not np.isnan(row["error_deg"]):
            heat[r_i, c_i]  = row["error_deg"]
            annot[r_i, c_i] = (f"{row['error_deg']:.2f}°\n"
                                f"±{row['std_deg']:.2f}")
        else:
            annot[r_i, c_i] = "—"

    n = len(all_results)
    fig, ax = plt.subplots(figsize=(15, 8))
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#555555")
    sns.heatmap(np.ma.masked_invalid(heat), annot=annot, fmt="",
                cmap=cmap, vmin=0, vmax=2.0,
                linewidths=1.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Erro médio (graus)"},
                annot_kws={"fontsize": 9, "color": "white", "weight": "bold"})
    for tid in CALIBRATION_TARGET_IDS:
        r_i, c_i = (tid - 1) // 7, (tid - 1) % 7
        ax.add_patch(plt.Rectangle((c_i, r_i), 1, 1,
                                   fill=False, edgecolor="red", lw=2.5))
    ax.set_title(f"Erro Angular Agregado — {n} Sujeitos (média ± DP entre sujeitos)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Coluna (x)", fontsize=12)
    ax.set_ylabel("Linha (y)",  fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_agregado.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Mapa de calor agregado salvo: heatmap_agregado.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo: segmentação v2 + calibração + relatório"
    )
    parser.add_argument(
        "--subjects", "-s", nargs="*", metavar="SUBJ",
        help="Lista de sujeitos a processar (omitir = todos)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Reprocessar segmentação mesmo que o CSV de cache já exista"
    )
    parser.add_argument(
        "--data_dir", default=DATA_DIR,
        help=f"Diretório raiz dos dados (padrão: {DATA_DIR})"
    )
    parser.add_argument(
        "--output_dir", default=OUTPUT_DIR,
        help=f"Diretório de saída (padrão: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_global = time.time()

    # ── Descoberta ─────────────────────────────────────────────────────────
    subjects = discover_subjects(args.data_dir, args.subjects)
    print(f"\n[Pipeline] {len(subjects)} sujeito(s) encontrados")
    print(f"[Pipeline] Saída: {args.output_dir}\n")

    all_results: List[dict] = []

    for subj, session_dirs in subjects.items():
        print(f"\n{'─'*60}")
        print(f"  Sujeito: {subj}   ({len(session_dirs)} sessões de fixação)")
        subj_out = os.path.join(args.output_dir, subj)
        os.makedirs(subj_out, exist_ok=True)

        # ── Etapa 1: Segmentação ────────────────────────────────────────
        print("  [1/3] Segmentação:")
        seg_csvs: List[str] = []
        for sess_dir in session_dirs:
            csv_path = run_segmentation(sess_dir, subj_out, force=args.force)
            if csv_path:
                seg_csvs.append(csv_path)

        if not seg_csvs:
            print("  [SKIP] Nenhum CSV gerado.")
            continue

        # ── Etapa 2: Calibração ─────────────────────────────────────────
        print("  [2/3] Calibração:")
        df_targets = load_and_aggregate(seg_csvs)
        if df_targets.empty:
            print("        Sem detecções válidas.")
            continue

        n_tgt = len(df_targets)
        print(f"        {n_tgt}/35 alvos com dados")

        result = calibrate(df_targets)
        if result is None:
            print(f"        Alvos de treino insuficientes "
                  f"(mínimo {MIN_CALIB_TARGETS}).")
            continue

        df_cal, model_x, model_y = result

        mean_err  = float(df_cal["error_deg"].mean())
        std_err   = float(df_cal["error_deg"].std())
        train_err = float(df_cal[df_cal["split"] == "train"]["error_deg"].mean())
        test_err  = float(df_cal[df_cal["split"] == "test"]["error_deg"].mean()
                          if (df_cal["split"] == "test").any() else np.nan)

        status_sym = "✓" if mean_err < 1.0 else "✗"
        print(f"        Erro médio geral: {mean_err:.3f}° ± {std_err:.3f}°  {status_sym}")
        print(f"        Treino:  {train_err:.3f}°   "
              f"Teste (held-out): {test_err:.3f}°")

        # ── Etapa 3: Salvamento ─────────────────────────────────────────
        print("  [3/3] Salvando resultados:")
        df_cal.to_csv(os.path.join(subj_out, "calibration_results.csv"),
                      index=False)

        hmap_path = os.path.join(subj_out, "heatmap.png")
        save_heatmap(df_cal,
                     f"{subj}  —  Erro: {mean_err:.2f}° ± {std_err:.2f}°",
                     hmap_path)

        scatter_path = os.path.join(subj_out, "scatter_calib.png")
        save_scatter(df_cal,
                     f"{subj}  —  Previsto vs Real",
                     scatter_path)

        print(f"        calibration_results.csv  ✓")
        print(f"        heatmap.png               ✓")
        print(f"        scatter_calib.png         ✓")

        all_results.append({
            "subject":   subj,
            "mean_deg":  mean_err,
            "std_deg":   std_err,
            "train_deg": train_err,
            "test_deg":  test_err,
            "n_targets": n_tgt,
            "df":        df_cal,
        })

    # ── Relatório final ──────────────────────────────────────────────────
    print_summary(all_results)

    if all_results:
        # CSV aggregado
        summary_rows = [{
            "sujeito":          r["subject"],
            "erro_medio_deg":   round(r["mean_deg"],  3),
            "desvio_padrao_deg":round(r["std_deg"],   3),
            "erro_treino_deg":  round(r["train_deg"], 3),
            "erro_teste_deg":   round(r["test_deg"],  3),
            "n_alvos":          r["n_targets"],
        } for r in all_results]

        # Adiciona linha global
        global_mean = float(np.mean([r["mean_deg"] for r in all_results]))
        global_std  = float(np.std( [r["mean_deg"] for r in all_results]))
        summary_rows.append({
            "sujeito": "GLOBAL",
            "erro_medio_deg":    round(global_mean, 3),
            "desvio_padrao_deg": round(global_std,  3),
            "erro_treino_deg":   np.nan,
            "erro_teste_deg":    np.nan,
            "n_alvos":           np.nan,
        })
        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(args.output_dir, "summary_final.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n  Tabela resumo: {summary_path}")

        # Mapa de calor agregado
        save_aggregate_heatmap(all_results, args.output_dir)

    elapsed = time.time() - t_global
    print(f"\n[Pipeline] Concluído em {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
