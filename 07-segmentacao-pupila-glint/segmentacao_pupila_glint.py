#!/usr/bin/env python3
"""
Script Completo: Detección Pupila/Glints con Validación ROBUSTA de Parpadeo en
Modo Interactivo (Tecla 'N') y Grabación de Video de Salida,
además de la exportación de datos a CSV.
"""

import os
import argparse
import collections
import math
import cv2
import numpy as np
import csv # Importación necesaria para manejar archivos CSV
import pandas as pd
import pandas as pd
from scipy.optimize import least_squares
from collections import deque


# === CLASE PARA EDICIÓN MANUAL ===
class ManualEditor:
    def __init__(self, window_name):
        self.window_name = window_name
        self.active = False
        self.clicks = []
        self.frame_type = None
        self.done = False
        self.w_sub = 0
        self.h_sub = 0
        
    def start_editing(self, frame_type, w_sub, h_sub):
        self.active = True
        self.clicks = []
        self.frame_type = frame_type
        self.done = False
        self.w_sub = w_sub
        self.h_sub = h_sub
        print(f"\n[EDIT] MODO EDICIÓN ({frame_type.upper()})")
        print("  - Click 1: Centro Pupila")
        if frame_type == 'brilhante':
            print("  - Click 2: Centro Glint")
        else:
            print("  - Click 2: Glint 1")
            print("  - Click 3: Glint 2")
        print("  - Tecla 'P': Marcar como Parpadeo")
        print("  - Tecla 'Esc': Cancelar")

    def mouse_callback(self, event, x, y, flags, param):
        if not self.active or self.done:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Verificar si el click es en el cuadrante superior izquierdo
            if x < self.w_sub and y < self.h_sub:
                real_x = int(x * 3)
                real_y = int(y * 2)
                
                self.clicks.append((real_x, real_y))
                print(f"[EDIT] Punto {len(self.clicks)}: ({real_x}, {real_y})")
                
                required_clicks = 2 if self.frame_type == 'brilhante' else 3
                if len(self.clicks) >= required_clicks:
                    self.done = True
                    print("[EDIT] Puntos completados.")

def update_csv_row(df, filename, status, px, py, gx, gy, vx, vy):
    if df is not None:
        mask = df['file_name'] == filename
        if mask.any():
            df.loc[mask, 'status'] = status
            df.loc[mask, 'pupil_x'] = float(f"{px:.2f}")
            df.loc[mask, 'pupil_y'] = float(f"{py:.2f}")
            df.loc[mask, 'glint_x'] = float(f"{gx:.2f}")
            df.loc[mask, 'glint_y'] = float(f"{gy:.2f}")
            df.loc[mask, 'vector_x'] = float(f"{vx:.2f}")
            df.loc[mask, 'vector_y'] = float(f"{vy:.2f}")
            print(f"[CSV] Actualizado {filename} | Status: {status}")

# === CONSTANTES DE VISUALIZACIÓN ===
COLOR_PUPILA = (255, 0, 0)        # Azul
COLOR_GLINT_1_DARK = (0, 0, 255)  # Rojo
COLOR_GLINT_2_DARK = (0, 255, 255)# Amarillo
COLOR_GLINT_BRIGHT = (0, 255, 255)# Amarillo
COLOR_TEXTO_MOSAICO = (0, 255, 0) 
COLOR_TEXTO_CONTEO_NORMAL = (255, 255, 0) 
COLOR_TEXTO_CONTEO_CRITICO = (0, 255, 255)

# ---------- Parámetros ----------
MIN_BRIGHT_PIXEL_VALUE = 200 
MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL = 2000 # 1500 15000 2745 600 7000 200 8000

GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL = 0.90
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL = 0.95
GLINT_MIN_AREA = 0
GLINT_MAX_AREA = 500
GLINT_CIRCULARITY_THRESHOLD = 0.60
GLINT_RADIUS_DARK = 2
GLINT_RADIUS_BRIGHT = 3
PUPIL_CENTER_RADIUS = 3

bright_circularidad_min = 0.35

# --- VARIABLES GLOBALES DE CONTROL ---
AUTO_MODE = False # Inicia en modo manual por defecto

# -------------------------------------------------------------------
## Funciones y Clases de Ayuda
# -------------------------------------------------------------------

def _dist(a, b):
    """Calcula la distancia euclidiana entre dos puntos."""
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _calculate_midpoint(p1, p2):
    """Calcula el punto medio entre dos coordenadas (x1, y1) y (x2, y2)."""
    if p1 is None or p2 is None:
        # En el contexto del CSV, si no hay puntos, la lógica de guardado
        # debería establecer 0, pero la función helper puede retornar (-1,-1)
        return -1.0, -1.0 
    mx = (p1[0] + p2[0]) / 2.0
    my = (p1[1] + p2[1]) / 2.0
    return mx, my

def draw_cross_full(image, center, color, radius=5):
    """Dibuja un círculo y una cruz que cruza todo el marco."""
    if center is None or len(center) != 2:
        return
    H, W = image.shape[:2] 
    x, y = int(center[0]), int(center[1])
    if 0 <= x < W and 0 <= y < H:
        cv2.line(image, (0, y), (W, y), color, 1) 
        cv2.line(image, (x, 0), (x, H), color, 1)
        cv2.circle(image, (x, y), radius, color, -1)

class CircleFitter:
    @staticmethod
    def fit_circle(points):
        """Ajusta un círculo a un conjunto de puntos usando mínimos cuadrados."""
        if points is None or len(points) < 3:
            return None
        x = points[:, 0].astype(float)
        y = points[:, 1].astype(float)

        def residuals(params, x_data, y_data):
            cx, cy, r = params
            return np.sqrt((x_data - cx) ** 2 + (y_data - cy) ** 2) - r

        cx0 = x.mean()
        cy0 = y.mean()
        r0 = np.mean(np.sqrt((x - cx0) ** 2 + (y - cy0) ** 2))
        try:
            res = least_squares(residuals, [cx0, cy0, r0], args=(x, y), bounds=([0, 0, 1], np.inf), xtol=1e-3)
            cx, cy, r = res.x
            return (float(cx), float(cy), float(r))
        except Exception:
            return None

class PupilTypeClassifier:
    @staticmethod
    def classify_by_bright_pixels(img_gray, min_bright_val, min_pixel_count):
        """Clasifica el marco como 'escura' o 'brilhante' basado en el conteo de píxeles brillantes."""
        if img_gray is None:
            return 'desconhecido', {}
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        bright_pixel_count = int(np.sum(hist[min_bright_val:256]))
        metrics = {'bright_pixel_count': bright_pixel_count}
        if bright_pixel_count >= min_pixel_count:
            return 'brilhante', metrics
            #return 'escura', metrics
        else:
            return 'escura', metrics
            #return 'brilhante', metrics

class GlintDetector:
    @staticmethod
    def detect_glints(img_gray, roi_mask=None, is_dark_pupil_frame=False, par_idx=0, is_pre_cleaning=False):
        """Detecta los glints (reflejos) en el ojo."""
        if img_gray is None:
            return [], np.zeros_like(img_gray) if img_gray is not None else None, np.zeros_like(img_gray) if img_gray is not None else None 
        img_to_process = img_gray.copy()
        if roi_mask is not None:
            if np.any(roi_mask):
                img_to_process = cv2.bitwise_and(img_gray, img_gray, mask=roi_mask)
        
        min_val, max_val, _, _ = cv2.minMaxLoc(img_to_process)
        if max_val < 50:
            return [], img_to_process, np.zeros_like(img_to_process)
            
        # Si es pre-limpieza para el ROI, usamos un umbral más bajo para atrapar el "halo"
        if is_pre_cleaning:
            glint_thresh = 180 # Umbral agresivo para atrapar todo el brillo
        else:
            threshold_perc = GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL if is_dark_pupil_frame else GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL
            glint_thresh = int(max_val * threshold_perc)
            glint_thresh = max(100, min(240, glint_thresh)) 

        _, thresh = cv2.threshold(img_to_process, glint_thresh, 255, cv2.THRESH_BINARY)
        
        kernel_close = np.ones((1,1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        kernel_open = np.ones((2,2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        contours_pre, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        MERGED_GLINT_MIN_AREA = 15 
        MERGED_GLINT_MAX_AREA = GLINT_MAX_AREA * 2 
        run_watershed = False
        
        if is_dark_pupil_frame:
            if len(contours_pre) == 1:
                area = cv2.contourArea(contours_pre[0])
                if MERGED_GLINT_MIN_AREA < area < MERGED_GLINT_MAX_AREA:
                    run_watershed = True
        
        if run_watershed:
            opening = thresh
            kernel_dilate = np.ones((3,3), np.uint8)
            sure_bg = cv2.dilate(opening, kernel_dilate, iterations=1)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg) 
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            img_bgr_for_watershed = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
            markers_final = cv2.watershed(img_bgr_for_watershed, markers.copy())
            new_mask = np.zeros_like(thresh)
            new_mask[markers_final > 1] = 255
            thresh = new_mask

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glints = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if GLINT_MIN_AREA < area < GLINT_MAX_AREA:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter * perimeter)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        glints.append({'center': (cx, cy), 'area': area, 'circularity': circularity})
        
        # [DEBUG] Info General Glints
        if len(glints) > 0 and not AUTO_MODE:
            print(f"info general sobre los glints del frame {par_idx if par_idx else 'None'} tiene {len(glints)}")
            for g in glints:
                print(f"el area es {g['area']:.1f} de los glint en el frame {par_idx if par_idx else 'None'} y debe estar entre {GLINT_MIN_AREA} e {GLINT_MAX_AREA}")
        
        return glints, img_to_process, thresh

class GlintSelector:
    @staticmethod
    def select_glints_bright(pupil_center, glint_candidates, pupil_mask=None):
        """Selecciona el glint más probable para la pupila brillante (uno solo)."""
        if not pupil_center or not glint_candidates:
            return []
        
        # --- Lógica de Filtrado por Máscara (glint debe estar fuera de la pupila) ---
        filtered_candidates = []
        if pupil_mask is not None and np.any(pupil_mask):
            for g in glint_candidates:
                x, y = g['center']
                ix, iy = int(x), int(y)
                if 0 <= iy < pupil_mask.shape[0] and 0 <= ix < pupil_mask.shape[1]:
                    if pupil_mask[iy, ix] == 0: # 0 significa fuera de la máscara (fondo)
                        filtered_candidates.append(g)
            
            if filtered_candidates:
                glint_candidates = filtered_candidates
        
        if not glint_candidates:
            return []

        # --- FILTRO DE CIRCULARIDAD MÍNIMA ---
        C_MIN = 0.70
        glint_candidates = [g for g in glint_candidates if g.get('circularity', 0) >= C_MIN]
        
        if not glint_candidates:
            return []

        # Ronda 1: Ordenar por Circularidad (del más redondo al menos redondo)
        glint_candidates.sort(key=lambda g: -g.get('circularity', 0))

        # Pasar a la final solo a los 2 mejores
        finalists = glint_candidates[:2]
        
        if len(finalists) <= 1:
            return finalists

        # Ronda 2 (Final): De los 2 finalistas, ordenar por cercanía a la pupila
        px, py = pupil_center
        finalists.sort(key=lambda g: (g['center'][0] - px)**2 + (g['center'][1] - py)**2)
        
        # El ganador es el primero (el más cercano)
        return finalists[:1]

    @staticmethod
    def select_glints_dark(pupil_center, glint_candidates, frame_num=0):
        """Selecciona los dos glints más cercanos (para pupila oscura)."""
        if not pupil_center or not glint_candidates:
            return []
        
        # 1. Glint más cercano a la pupila
        def distance_score(g):
            return _dist(g['center'], pupil_center)
        glint_candidates.sort(key=distance_score)
        glint_1 = glint_candidates[0]
        
        if len(glint_candidates) == 1:
            return [glint_1]

        # 2. Glint más cercano al Glint 1
        remaining_candidates = glint_candidates[1:]
        def proximity_to_g1(g):
            return _dist(g['center'], glint_1['center'])
        
        remaining_candidates.sort(key=proximity_to_g1)
        glint_2 = remaining_candidates[0]
        
        return [glint_1, glint_2]       

# =================================================================================
# CLASE PUPILSEGMENTER (Lógica de Segmentación y Veto)
# =================================================================================
class PupilSegmenter:
    def __init__(self):
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)
        self.show_debug_windows = False

    def _to_bgr(self, gray_img):
        if gray_img is None:
            return None
        if gray_img.ndim == 2:
            return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        return gray_img

    def _null_result(self, escura_gray, brilhante_gray, diff, roi_mask, blink_detected=True): 
        h, w = escura_gray.shape[:2]
        zero_mask = np.zeros((h, w), dtype=np.uint8)
        
        return {
            'escura_gray': escura_gray, 'brilhante_gray': brilhante_gray, 
            'diferenca': diff if diff is not None else zero_mask, 
            'roi_mask': roi_mask if roi_mask is not None else zero_mask,
            'resultado_escura_borda': self._to_bgr(escura_gray), 'pupilas_escuras': [], 
            'glints_escura_frame': [], 'pupil_mask_escura': zero_mask,
            'resultado_brilhante_borda': self._to_bgr(brilhante_gray), 'pupilas_brillantes': [], 
            'glints_brilhante_frame': [], 'pupil_mask_brilhante': zero_mask,
            'glint_roi_mask': zero_mask, 
            'escura_bitwise_and': zero_mask, 'escura_thresh': zero_mask,
            'brilhante_bitwise_and': zero_mask, 'brilhante_thresh': zero_mask,
            'morph_mask': zero_mask,
            'blink_detected': blink_detected 
        }

    def refine_roi(self, escura_gray, brilhante_gray, current_frame_index, fallback_roi_mask=None):
        diff = None
        roi_mask = None
        
        morph_mask = None
        if fallback_roi_mask is not None and np.any(fallback_roi_mask):
            roi_mask = fallback_roi_mask.copy()
            morph_mask = roi_mask.copy() # En fallback, asumimos que la máscara ya está procesada
            diff = cv2.absdiff(brilhante_gray, escura_gray)
            #diff_blurred = cv2.GaussianBlur(diff, (9, 9), 0) 
        else:
            # Usamos subtract (Brilhante - Escura) para resaltar solo el aumento de brillo de la pupila
            diff = cv2.absdiff(brilhante_gray, escura_gray) 
            diff_blurred = cv2.GaussianBlur(diff, (9, 9), 0)
            _, roi_mask = cv2.threshold(diff_blurred, 120, 255, cv2.THRESH_BINARY) # min 120
        
            if roi_mask is None or not np.any(roi_mask):
                return None, None, None 

            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.kernel_medium, iterations=2)
            roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.kernel_medium, iterations=1)
            morph_mask = roi_mask.copy() # Guardar estado después de morfología

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_mask, 8, cv2.CV_32S)
            if num_labels > 1: 
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_label_index = np.argmax(areas)
                largest_label = largest_label_index + 1
                new_mask = np.zeros_like(roi_mask)
                new_mask[labels == largest_label] = 255
                roi_mask = new_mask

        if roi_mask is None or not np.any(roi_mask):
            return None, None, None
            
        return roi_mask, diff, morph_mask

    def remove_glints_from_image(self, img_gray, glints_info):
        if not glints_info:
            return img_gray.copy()
            
        # Crear una máscara para los glints
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        for glint in glints_info:
            center = glint['center']
            ix, iy = int(center[0]), int(center[1])
            # Aumentamos el radio (factor 2.5) para cubrir el halo blanco del glint
            radius = int(max(3, np.sqrt(glint['area'] / np.pi) * 2.5))
            cv2.circle(mask, (ix, iy), radius, 255, -1)
            
        # Aplicar Inpainting para rellenar los huecos de los glints
        img_no_glints = cv2.inpaint(img_gray, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        return img_no_glints

    # === SEGMENT_PUPIL_DARK (Lógica de Veto) ===
    def segment_pupil_dark(self, escura_gray_no_glints, roi_mask, frame_original, frame_idx):
        pupil_mask = None
        result_img = frame_original.copy()
        pupil_info_list = []
        
        _, thresh_inv = cv2.threshold(escura_gray_no_glints, 50, 255, cv2.THRESH_BINARY_INV)
        thresh_inv_roi = cv2.bitwise_and(thresh_inv, thresh_inv, mask=roi_mask)

        opening = cv2.morphologyEx(thresh_inv_roi, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        sure_bg = cv2.dilate(opening, self.kernel_small, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        if dist_transform is None:
            return result_img, np.zeros_like(escura_gray_no_glints), [], True 

        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.absdiff(sure_bg, sure_fg) 
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        if np.max(markers) > 1:
            markers_final = cv2.watershed(frame_original.copy(), markers.copy())
            
            # [VISUALIZACION] Pintar bordes de watershed en result_img (Excluyendo bordes de la imagen)
            border_mask = (markers_final == -1)
            # Eliminar los bordes de la imagen de la máscara
            border_mask[0, :] = False
            border_mask[-1, :] = False
            border_mask[:, 0] = False
            border_mask[:, -1] = False
            result_img[border_mask] = [0, 0, 255] # Rojo para bordes internos

            mask_all = np.zeros_like(escura_gray_no_glints)
            mask_all[markers_final > 1] = 255
            pupil_mask = mask_all.copy()

            contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                # --- VETO 1: Aspect Ratio ---
                aspect_ratio = float(h) / w if w > 0 else 0
                if not AUTO_MODE:
                    print(f"[DEBUG] [DARK] frame {frame_idx} 1er filtro pupila, Aspect Ratio: {aspect_ratio:.2f} debe ser entre 0.73 e 1.2")
                if aspect_ratio < 0.65 or aspect_ratio > 1.5: 
                    continue 
                
                area_min = 500
                #if frame_idx == 276:
                #    area_min = 217
                #else:
                #    area_min = 2000    

                if not AUTO_MODE:
                    print(f"[DEBUG] [DARK] frame {frame_idx} 2do filtro pupila, de Area: {area:.1f} entre {area_min} y 11000")
                if area_min < area < 11000:
                    
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        if not AUTO_MODE:
                            print(f"[DEBUG] [DARK] frame {frame_idx} 3er filtro pupila, Circularidad: {circularity:.2f} debe ser mayor que 0.32")
                        if circularity > 0.2: 
                            points = cnt.squeeze()
                            if points.ndim == 1 or points.shape[0] < 3:
                                continue
                            fit = CircleFitter.fit_circle(points)
                            
                            if fit is not None:
                                cx, cy, r = fit
                                
                                # --- VETO 2: Radio Mínimo ---
                                if not AUTO_MODE:
                                    print(f"[DEBUG] [DARK] frame {frame_idx} 4to filtro pupila, Radio: {r:.2f} debe ser mayor que 6.0")
                                if r < 6:
                                    continue
                                
                                center = (cx, cy)
                                pupil_info_list.append({
                                    'center': center,
                                    'radius': r,
                                    'segmentada': True,
                                    'contour': cnt,
                                    'fit_circle': fit
                                })
        # === LÓGICA FINAL Y DEBUG ===
        is_blink = (len(pupil_info_list) == 0)
        
        if not AUTO_MODE:
            print(f"[DEBUG] [DARK] cantidad de Pupilas detectadas: {len(pupil_info_list)} en frame {frame_idx}. Blink: {is_blink}")

        # Para desactivar el debug, cambia DEBUG_PASO_A_PASO_LOCAL a False
        DEBUG_PASO_A_PASO_LOCAL = False 
        if DEBUG_PASO_A_PASO_LOCAL and not AUTO_MODE:
            # 0. Imagen de entrada sin glints
            cv2.imshow("DEBUG DARK 0: Gray No Glints", escura_gray_no_glints)
            
            # 1. Mostrar Umbral
            cv2.imshow("DEBUG DARK 1: Thresh Inv ROI", thresh_inv_roi)
            if dist_transform is not None:
                dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("DEBUG DARK 2: Distance Transform", dist_vis)
            if 'markers_final' in locals():
                markers_vis = np.uint8(cv2.normalize(markers_final, None, 0, 255, cv2.NORM_MINMAX))
                markers_vis = cv2.applyColorMap(markers_vis, cv2.COLORMAP_JET)
                cv2.imshow("DEBUG DARK 3: Watershed Markers", markers_vis)
            cv2.imshow("DEBUG DARK 4: Resultado Final", result_img)
            # cv2.waitKey(0) # Eliminado para que se actualicen todas a la vez al final

        return result_img, pupil_mask, pupil_info_list, is_blink

    # === SEGMENT_PUPIL_BRIGHT (Lógica de Veto) ===
    def segment_pupil_bright(self, brilhante_gray_no_glints, roi_mask, frame_original, frame_idx):
        pupil_mask = None
        result_img = frame_original.copy()
        pupil_info_list = []

        roi_no_glints = cv2.bitwise_and(brilhante_gray_no_glints, brilhante_gray_no_glints, mask=roi_mask)

        max_val = 0
        try:
            min_v, max_v, _, _ = cv2.minMaxLoc(roi_no_glints, mask=roi_mask)
            max_val = max_v
        except:
            max_val = 0

        # Aplicar CLAHE para mejorar contraste local entre pupila e iris
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi_enhanced = clahe.apply(roi_no_glints)

        # Usar OTSU para umbralización automática adaptativa
        _, thresh = cv2.threshold(roi_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Limpieza morfológica suave (sin erosión agresiva)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_small, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel_small, iterations=1)

        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        sure_bg = cv2.dilate(opening, self.kernel_small, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        if dist_transform is None:
             return result_img, np.zeros_like(brilhante_gray_no_glints), [], True

        dist_max = dist_transform.max() if dist_transform is not None else 0
        if dist_max > 0:
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_max, 255, 0)
        else:
            sure_fg = np.zeros_like(brilhante_gray_no_glints)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.absdiff(sure_bg, sure_fg) 
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        if np.max(markers) > 1:
            markers_final = cv2.watershed(frame_original.copy(), markers.copy())
            
            # [VISUALIZACION] Pintar bordes de watershed en result_img (Excluyendo bordes de la imagen)
            border_mask = (markers_final == -1)
            # Eliminar los bordes de la imagen de la máscara
            border_mask[0, :] = False
            border_mask[-1, :] = False
            border_mask[:, 0] = False
            border_mask[:, -1] = False
            result_img[border_mask] = [0, 0, 255] # Rojo para bordes internos

            mask_all = np.zeros_like(brilhante_gray_no_glints)
            mask_all[markers_final > 1] = 255
            pupil_mask = mask_all.copy()

            contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(f"Esta es la cantida<d de contornos encontrados por watershed: {len(contours)}")
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                
                aspect_ratio = float(h) / w if w > 0 else 0
                if not AUTO_MODE:
                    print(f"[DEBUG] [BRIGHT] frame {frame_idx} cand {i+1} 1er filtro pupila, Aspect Ratio: {aspect_ratio:.2f} debe ser entre 0.40 e 1.7")
                if aspect_ratio < 0.40 or aspect_ratio > 1.8: 
                    continue
                
                if not AUTO_MODE:
                    print(f"[DEBUG] [BRIGHT] frame {frame_idx} cand {i+1}, 2do filtro pupila, de Area: {area:.1f} entre 700 y 11000")
                area_min_bright = 50
                #if frame_idx == 81:
                #    area_min_bright = 200

                if area_min_bright < area < 20000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        
                        global bright_circularidad_min
                        #if frame_idx == 116 or frame_idx == 119 or frame_idx == 121 or frame_idx == 123 or frame_idx == 125:
                        #    bright_circularidad_min = 0.1
                        #elif frame_idx == 370:
                        #    bright_circularidad_min = 0.7
                        #else:
                        bright_circularidad_min = 0.20    

                        if not AUTO_MODE:
                            print(f"[DEBUG] [BRIGHT] frame {frame_idx} cand {i+1}, 3er filtro pupila, Circularidad: {circularity:.2f} debe ser mayor que {bright_circularidad_min}")
                        if circularity > bright_circularidad_min:
                            points = cnt.squeeze()
                            if points.ndim == 1 or points.shape[0] < 3:
                                continue
                            fit = CircleFitter.fit_circle(points)
                            if fit is not None:
                                cx, cy, r = fit
                                if not AUTO_MODE:
                                    print(f"[DEBUG] [BRIGHT] frame {frame_idx} cand {i+1}, 4to filtro pupila, Radio: {r:.2f} debe ser mayor que 4.0")
                                if r < 4:
                                    continue

                                center = (cx, cy)
                                pupil_info_list.append({
                                    'center': center,
                                    'radius': r,
                                    'segmentada': True,
                                    'contour': cnt,
                                    'fit_circle': fit
                                })

        # ORDENAR candidatos por ÁREA (de mayor a menor) para elegir la pupila real
        if len(pupil_info_list) > 1:
            pupil_info_list.sort(key=lambda p: cv2.contourArea(p['contour']), reverse=True)
            if not AUTO_MODE:
                print(f"[DEBUG] [BRIGHT] Ordenados {len(pupil_info_list)} candidatos por área")
        # === LÓGICA FINAL Y DEBUG ===
        if len(pupil_info_list) > 0 and frame_idx == 370:
            is_blink = False
        elif frame_idx == 370:
            is_blink = True
        elif len(pupil_info_list) > 0:
            is_blink = False
        else:
            is_blink = True

        if not AUTO_MODE:
            print(f"[DEBUG] [BRIGHT] cantidad de Pupilas detectadas: {len(pupil_info_list)} en frame {frame_idx}. Blink: {is_blink}")

        # Para desactivar el debug, cambia DEBUG_PASO_A_PASO_LOCAL a False
        DEBUG_PASO_A_PASO_LOCAL = False 
        if DEBUG_PASO_A_PASO_LOCAL and not AUTO_MODE:
            # 0. Imagen de entrada sin glints
            cv2.imshow("DEBUG BRIGHT 0: Gray No Glints", brilhante_gray_no_glints)
            
            # 1. Mostrar Umbral
            cv2.imshow("DEBUG BRIGHT 1: Thresh ROI", thresh)
            if dist_transform is not None:
                dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("DEBUG BRIGHT 2: Distance Transform", dist_vis)
            if 'markers_final' in locals():
                markers_vis = np.uint8(cv2.normalize(markers_final, None, 0, 255, cv2.NORM_MINMAX))
                markers_vis = cv2.applyColorMap(markers_vis, cv2.COLORMAP_JET)
                cv2.imshow("DEBUG BRIGHT 3: Watershed Markers", markers_vis)
            cv2.imshow("DEBUG BRIGHT 4: Resultado Final", result_img)
            # cv2.waitKey(0) # Eliminado para que se actualicen todas a la vez al final

        return result_img, pupil_mask, pupil_info_list, is_blink

    def segment_pair(self, frame_escura, frame_brilhante, par_idx, target_type=None, fallback_roi_mask=None):
        brilhante_no_glints = None
        if frame_escura is None or frame_brilhante is None:
            return None

        escura_gray = cv2.cvtColor(frame_escura, cv2.COLOR_BGR2GRAY)
        brilhante_gray = cv2.cvtColor(frame_brilhante, cv2.COLOR_BGR2GRAY)

        # --- MEJORA: Detectar y quitar glints ANTES del ROI para que sea más redondo ---
        # Usamos is_pre_cleaning=True para ser más agresivos y atrapar el halo
        glints_e_pre, _, _ = GlintDetector.detect_glints(escura_gray, None, is_dark_pupil_frame=True, par_idx=par_idx, is_pre_cleaning=True)
        glints_b_pre, _, _ = GlintDetector.detect_glints(brilhante_gray, None, is_dark_pupil_frame=False, par_idx=par_idx, is_pre_cleaning=True)
        
        escura_gray_clean = self.remove_glints_from_image(escura_gray, glints_e_pre)
        brilhante_gray_clean = self.remove_glints_from_image(brilhante_gray, glints_b_pre)

        # Usamos las imágenes LIMPIAS y cv2.subtract para el ROI
        roi_mask, diff, morph_mask = self.refine_roi(escura_gray_clean, brilhante_gray_clean, par_idx, fallback_roi_mask)
        
        if roi_mask is None or not np.any(roi_mask):
            return self._null_result(escura_gray, brilhante_gray, diff, roi_mask, blink_detected=True)
        
        kernel_glint = np.ones((32, 32), np.uint8) # de 15 x 15 antes depois 23 x 23
        glint_roi_mask = cv2.dilate(roi_mask, kernel_glint, iterations=4) # 2 iteraciones antes despues 3

        # Inicializamos
        blink_escura = False
        blink_brilhante = False
        escura_resultado = frame_escura.copy()
        brilhante_resultado = frame_brilhante.copy()
        pupil_escura_info = []
        pupil_brilhante_info = []
        pupil_mask_escura = np.zeros_like(escura_gray)
        pupil_mask_brilhante = np.zeros_like(brilhante_gray)
        glints_escura, glints_brilhante = [], []
        bitwise_escura, bitwise_brilhante = np.zeros_like(escura_gray), np.zeros_like(brilhante_gray)
        thresh_escura, thresh_brilhante = np.zeros_like(escura_gray), np.zeros_like(escura_gray) # Corregir inicialización de thresh_brilhante si es necesario
        
        # 1. EJECUTAR DETECCIONES (Lógica Normal o Fallback)
        if fallback_roi_mask is not None and target_type is not None:
            # MODO FALLBACK
            if target_type == 'escura':
                glints_escura, bitwise_escura, thresh_escura = GlintDetector.detect_glints(escura_gray, glint_roi_mask, is_dark_pupil_frame=True, par_idx=par_idx)
                escura_no_glints = self.remove_glints_from_image(escura_gray, glints_escura)
                escura_resultado, pupil_mask_escura, pupil_escura_info, blink_escura = self.segment_pupil_dark(escura_no_glints, roi_mask, frame_escura, par_idx)
                blink_brilhante = True 
            elif target_type == 'brilhante':
                glints_brilhante, bitwise_brilhante, thresh_brilhante = GlintDetector.detect_glints(brilhante_gray, glint_roi_mask, is_dark_pupil_frame=False, par_idx=par_idx)
                brilhante_no_glints = self.remove_glints_from_image(brilhante_gray, glints_brilhante)
                brilhante_resultado, pupil_mask_brilhante, pupil_brilhante_info, blink_brilhante = self.segment_pupil_bright(brilhante_no_glints, roi_mask, frame_brilhante, par_idx)
                blink_escura = True
        else:
            # MODO NORMAL (Ambos)
            glints_escura, bitwise_escura, thresh_escura = GlintDetector.detect_glints(escura_gray, glint_roi_mask, is_dark_pupil_frame=True, par_idx=par_idx)
            glints_brilhante, bitwise_brilhante, thresh_brilhante = GlintDetector.detect_glints(brilhante_gray, glint_roi_mask, is_dark_pupil_frame=False, par_idx=par_idx)
            
            escura_no_glints = self.remove_glints_from_image(escura_gray, glints_escura)
            brilhante_no_glints = self.remove_glints_from_image(brilhante_gray, glints_brilhante)

            escura_resultado, pupil_mask_escura, pupil_escura_info, blink_escura = self.segment_pupil_dark(
                escura_no_glints, glint_roi_mask, frame_escura, par_idx
            )
            
            brilhante_resultado, pupil_mask_brilhante, pupil_brilhante_info, blink_brilhante = self.segment_pupil_bright(
                brilhante_no_glints, glint_roi_mask, frame_brilhante, par_idx
            )

        # 2. LOGICA DE CONTRADICCIÓN
        final_blink_detected = False
        
        # Caso A: Ambos dicen Blink -> Blink seguro
        if blink_escura and blink_brilhante:
            final_blink_detected = True
        else:
            # Lógica de Conflicto Simplificada (prioriza detección sobre parpadeo)
            if not blink_escura or not blink_brilhante:
                final_blink_detected = False # No es parpadeo si uno o ambos detectaron
            else:
                final_blink_detected = True # Caso teórico, debería ser atrapado arriba
                
        return {
            'escura_gray': escura_gray,
            'brilhante_gray': brilhante_gray,
            'diferenca': diff,
            'morph_mask': morph_mask,
            'roi_mask': roi_mask,
            'resultado_escura_borda': escura_resultado,
            'pupilas_escuras': pupil_escura_info,
            'glints_escura_frame': glints_escura,
            'pupil_mask_escura': pupil_mask_escura,
            'resultado_brilhante_borda': brilhante_resultado,
            'pupilas_brillantes': pupil_brilhante_info,
            'glints_brilhante_frame': glints_brilhante,
            'pupil_mask_brilhante': brilhante_no_glints,
            'escura_bitwise_and': bitwise_escura,
            'escura_thresh': thresh_escura,
            'brilhante_bitwise_and': bitwise_brilhante,
            'brilhante_thresh': thresh_brilhante,
            'glint_roi_mask': glint_roi_mask,
            'blink_detected': final_blink_detected
        }

# -------------------------------------------------------------------
## Función de Inicialización Silenciosa (Bootstrap)
# -------------------------------------------------------------------

def initial_roi_bootstrap(image_files, input_folder, seg, window_size):
    """Busca el primer par Dark/Bright exitoso para obtener una ROI inicial fiable."""
    print("[BOOTSTRAP INICIAL] Buscando el primer par Dark/Bright exitoso...")
    buffer = collections.deque(maxlen=2)
    
    for idx in range(len(image_files)):
        filename = image_files[idx]
        path = os.path.join(input_folder, filename)
        frame = cv2.imread(path)
        if frame is None: continue

        if frame.shape[1] != window_size[0] or frame.shape[0] != window_size[1]:
            frame = cv2.resize(frame, window_size)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tipo, _ = PupilTypeClassifier.classify_by_bright_pixels(frame_gray, MIN_BRIGHT_PIXEL_VALUE, MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL)
        buffer.append({'frame_data': frame, 'tipo': tipo, 'idx': idx + 1, 'filename': filename})
        
        if len(buffer) == 2:
            f2 = buffer[-1]
            f1 = buffer[0]
            tipo_f1 = f1['tipo']
            tipo_f2 = f2['tipo']
            
            if tipo_f1 != tipo_f2 and tipo_f1 != 'desconhecido' and tipo_f2 != 'desconhecido':
                frame_escura = f1['frame_data'] if tipo_f1 == 'escura' else f2['frame_data']
                frame_brilhante = f1['frame_data'] if tipo_f1 == 'brilhante' else f2['frame_data']
                
                # Usamos None en fallback_roi_mask para forzar la detección de una ROI nueva
                results = seg.segment_pair(frame_escura, frame_brilhante, par_idx=f2['idx'], target_type=None, fallback_roi_mask=None)
                
                if results and np.any(results['roi_mask']):
                    pupilas_e = results.get('pupilas_escuras', [])
                    pupilas_b = results.get('pupilas_brillantes', [])
                    
                    # Verificación estricta: debe haber segmentado al menos una pupila válida
                    if pupilas_e or pupilas_b:
                        print(f"[BOOTSTRAP INICIAL] ROI Mask obtenida exitosamente en el par F{f1['idx']}/F{f2['idx']}.")
                        return results['roi_mask']
                        
    print("[BOOTSTRAP INICIAL] Advertencia: No se encontró un par válido.")
    return None

def save_debug_images(output_folder, filename, results, tipo_f2, crosshair_img=None):
    """Guarda imágenes de depuración específicas para el reporte"""
    base_name = os.path.splitext(filename)[0]
    subfolder = os.path.join(output_folder, "processed_images")
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    
    prefix = 'brilhante' if tipo_f2 == 'brilhante' else 'escura'
    
    # 1. Imagen Original
    if tipo_f2 == 'brilhante':
        cv2.imwrite(os.path.join(subfolder, f"{base_name}_original_brilhante.png"), results['brilhante_gray'])
    else:
        cv2.imwrite(os.path.join(subfolder, f"{base_name}_original_escura.png"), results['escura_gray'])
        
    # 2. Diferencia (Thresholded)
    if results.get('roi_mask') is not None:
        cv2.imwrite(os.path.join(subfolder, f"{base_name}_diferencia.png"), results['roi_mask'])
        
    # 3. Watershed Result (Overlay) - Usamos el resultado con bordes dibujados por watershed
    if results.get(f'resultado_{prefix}_borda') is not None:
        cv2.imwrite(os.path.join(subfolder, f"{base_name}_watershed_overlay.png"), results[f'resultado_{prefix}_borda'])
    
    # 4. Contour (Imagen del frame con la circunferencia interpolada superpuesta)
    #    El usuario quiere: "imagen del frame pero superpuesta la circunferencia que se genera por interpolacion circular"
    
    # Recuperamos la lista de pupilas detectadas
    pupil_key = 'pupilas_brillantes' if tipo_f2 == 'brilhante' else 'pupilas_escuras'
    pupilas = results.get(pupil_key, [])
    
    # Usamos la imagen original como base (en BGR)
    if tipo_f2 == 'brilhante':
        contour_img = cv2.cvtColor(results['brilhante_gray'], cv2.COLOR_GRAY2BGR)
    else:
        contour_img = cv2.cvtColor(results['escura_gray'], cv2.COLOR_GRAY2BGR)
        
    if pupilas and pupilas[0].get('fit_circle'):
        cx, cy, r = pupilas[0]['fit_circle']
        # Dibujar circunferencia interpolada (Verde, grosor 2)
        cv2.circle(contour_img, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        # Dibujar centro
        cv2.circle(contour_img, (int(cx), int(cy)), 2, (0, 0, 255), -1)
        
    cv2.imwrite(os.path.join(subfolder, f"{base_name}_contorno.png"), contour_img)
        
    # 5. Crosshairs (Imagen con cruces)
    if crosshair_img is not None:
        cv2.imwrite(os.path.join(subfolder, f"{base_name}_crosshairs.png"), crosshair_img)

    # 6. Segmentación de Glints (Overlay de bordes de glints)
    #    "segmentação do glint na pupila brilhante" y "segmentação dos glints na pupila escura"
    
    # Glints Brilhante (SOLO SI EL FRAME ES TIPO BRILHANTE)
    if tipo_f2 == 'brilhante':
        if results.get('brilhante_thresh') is not None and results.get('brilhante_gray') is not None:
            glint_mask_b = results['brilhante_thresh']
            glint_vis_b = cv2.cvtColor(results['brilhante_gray'], cv2.COLOR_GRAY2BGR)
            
            # Encontrar contornos de los glints
            contours_g, _ = cv2.findContours(glint_mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Dibujar contornos en Azul Oscuro (BGR: 255, 0, 0)
            cv2.drawContours(glint_vis_b, contours_g, -1, (255, 0, 0), 1)
            
            cv2.imwrite(os.path.join(subfolder, f"{base_name}_glints_segmentation_bright.png"), glint_vis_b)
            
        # Limpieza: Asegurar que NO exista la imagen de dark para este frame
        wrong_file = os.path.join(subfolder, f"{base_name}_glints_segmentation_dark.png")
        if os.path.exists(wrong_file):
            os.remove(wrong_file)

    # Glints Escura (SOLO SI EL FRAME ES TIPO ESCURA)
    if tipo_f2 == 'escura':
        if results.get('escura_thresh') is not None and results.get('escura_gray') is not None:
            glint_mask_e = results['escura_thresh']
            glint_vis_e = cv2.cvtColor(results['escura_gray'], cv2.COLOR_GRAY2BGR)
            
            # Encontrar contornos de los glints
            contours_g, _ = cv2.findContours(glint_mask_e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Dibujar contornos en Rojo Oscuro (BGR: 0, 0, 139)
            cv2.drawContours(glint_vis_e, contours_g, -1, (0, 0, 139), 1)
            
            cv2.imwrite(os.path.join(subfolder, f"{base_name}_glints_segmentation_dark.png"), glint_vis_e)

        # Limpieza: Asegurar que NO exista la imagen de bright para este frame
        wrong_file = os.path.join(subfolder, f"{base_name}_glints_segmentation_bright.png")
        if os.path.exists(wrong_file):
            os.remove(wrong_file)
        
        cv2.imwrite(os.path.join(subfolder, f"{base_name}_glints_segmentation_dark.png"), glint_vis_e)

# -------------------------------------------------------------------
## Automatic Threshold Calculation (K-means)
# -------------------------------------------------------------------

def calculate_dynamic_threshold(image_files, input_folder, min_bright_val):
    """
    Calcula automaticamente el umbral MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL usando K-means (k=2).
    Analiza la distribución de conteos de píxeles brillantes en todas las imágenes.
    """
    print("\n[AUTO-CALIBRATION] Iniciando cálculo automático de umbral...")
    
    pixel_counts = []
    valid_files = []

    # 1. Recolectar datos
    for i, filename in enumerate(image_files):
        path = os.path.join(input_folder, filename)
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
            
        # Calcular conteo de píxeles brillantes
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        count = int(np.sum(hist[min_bright_val:256]))
        pixel_counts.append(count)
        valid_files.append(filename)
        
        if i % 10 == 0:
            print(f"\r[AUTO-CALIBRATION] Analizando frame {i+1}/{len(image_files)}...", end="")
    
    print(f"\r[AUTO-CALIBRATION] Analizados {len(pixel_counts)} frames.          ")

    if not pixel_counts:
        print("[AUTO-CALIBRATION] Error: No se pudieron leer imágenes.")
        return None

    data = np.array(pixel_counts)
    
    # 2. Implementación Simple de K-means (1D, k=2)
    # Inicialización: min y max como centroides iniciales
    centroids = np.array([np.min(data), np.max(data)], dtype=float)
    
    for _ in range(10): # 10 iteraciones suelen ser suficientes para 1D simple
        # Asignación
        distances = np.abs(data[:, np.newaxis] - centroids)
        labels = np.argmin(distances, axis=1)
        
        # Actualización
        new_centroids = np.array([data[labels == 0].mean(), data[labels == 1].mean()])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    # Identificar clusters (Low vs High)
    centroids.sort() # [Low, High]
    low_cluster_mean = centroids[0]
    high_cluster_mean = centroids[1]
    
    # Separar datos para encontrar bordes
    low_cluster_data = data[data <= (low_cluster_mean + high_cluster_mean) / 2]
    high_cluster_data = data[data > (low_cluster_mean + high_cluster_mean) / 2]
    
    if len(low_cluster_data) == 0 or len(high_cluster_data) == 0:
        print("[AUTO-CALIBRATION] Advertencia: No se encontró una distribución bimodal clara.")
        # Fallback: usar promedio simple o mantener valor por defecto
        return None

    max_of_min = np.max(low_cluster_data)
    min_of_max = np.min(high_cluster_data)
    
    # 3. Calcular Umbral (Punto medio entre los dos grupos)
    calculated_threshold = int((max_of_min + min_of_max) / 2)
    
    print(f"[AUTO-CALIBRATION] Cluster Oscuro (Mean): {low_cluster_mean:.1f} | Max: {max_of_min}")
    print(f"[AUTO-CALIBRATION] Cluster Brillante (Mean): {high_cluster_mean:.1f} | Min: {min_of_max}")
    print(f"[AUTO-CALIBRATION] UMBRAL CALCULADO: {calculated_threshold}")
    
    return calculated_threshold


def main(input_folder, loop_video=False): 
    WINDOW_WIDTH = 640 * 3
    WINDOW_HEIGHT = 480 * 2
    OUTPUT_FILENAME = "output_processed_sequence.avi"
    CSV_OUTPUT_FILENAME = "segmentation_data.csv"
    FOURCC = cv2.VideoWriter_fourcc(*'XVID') 
    
    global AUTO_MODE 
    
    if not os.path.exists(input_folder):
        print(f"[!] Carpeta no encontrada: {input_folder}")
        return
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.pgm')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    image_files.sort() 
    
    total_images = len(image_files)
    if total_images == 0:
        print(f"[ERROR] No se encontraron imágenes válidas en {input_folder}")
        return
    print(f"[i] Encontradas {total_images} imágenes en: {input_folder}")

    # --- AUTO CALIBRATION ---
    global MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL
    #calculated_thresh = calculate_dynamic_threshold(image_files, input_folder, MIN_BRIGHT_PIXEL_VALUE)
    calculated_thresh = None
    if calculated_thresh is not None:
        MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL = calculated_thresh
        print(f"[i] Usando UMBRAL DINÁMICO: {MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL}")
    else:
        print(f"[i] Usando UMBRAL POR DEFECTO: {MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL}")

    # --- OUTPUT FOLDER SETUP ---
    # Crear carpeta processed_{nombre_carpeta_entrada}
    input_folder_name = os.path.basename(os.path.normpath(input_folder))
    #output_folder_name = r"data/processed/subject_00120"
    output_folder_name = r"data/resultado_pupila_glint/subject10_P10"
    #output_folder_name = f"processed_{input_folder_name}"
    output_folder = os.path.join(os.getcwd(), output_folder_name)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[i] Carpeta de salida creada: {output_folder}")
    else:
        print(f"[i] Usando carpeta de salida existente: {output_folder}")

    # --- CSV DATA SETUP ---
    # Buscar CSV original en la carpeta de entrada
    csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv') and 'processed' not in f.lower()]
    original_csv_path = None
    df = None
    
    if csv_files:
        original_csv_path = os.path.join(input_folder, csv_files[0])
        print(f"[i] CSV original encontrado: {original_csv_path}")
        try:
            df = pd.read_csv(original_csv_path)
            # Asegurar que las columnas existan y sean de tipo FLOAT para evitar FutureWarnings
            for col in ['status', 'pupil_x', 'pupil_y', 'glint_x', 'glint_y', 'vector_x', 'vector_y']:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].astype(float)
        except Exception as e:
            print(f"[ERROR] Al cargar CSV: {e}")
            df = None
            
    if df is not None:
        # Check if we have smashed columns (semicolon issue)
        # Look for semicolons in ANY column name, not just if there's only 1 column
        if any(';' in str(col) for col in df.columns):
            print("[i] Detectado posible delimitador ';'. Recargando CSV...")
            try:
                df = pd.read_csv(original_csv_path, sep=';')
                print("[i] CSV recargado exitosamente con delimitador ';'")
            except Exception as e:
                print(f"[ERROR] Falló recarga con punto y coma: {e}")

        # Normalizar nombres de columnas (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        print(f"[DEBUG] Columnas en CSV: {list(df.columns)}")
        
        # Intentar identificar la columna de nombre de archivo
        possible_names = ['file_name', 'filename', 'FILE_NAME', 'FILENAME', 'image', 'Image', 'frame', 'Frame']
        found_col = None
        for name in possible_names:
            if name in df.columns:
                found_col = name
                break
        
        if found_col:
             print(f"[i] Columna de archivo detectada: '{found_col}'")
             if found_col != 'file_name':
                 df.rename(columns={found_col: 'file_name'}, inplace=True)
        else:
             print("[ERROR] No se encontró columna de nombre de archivo (file_name/filename/etc).")
             # Fallback: asumimos que se llaman 'file_name' si no existe, o creamos una vacía para evitar crash
             pass
    else:
        print("[WARN] No se encontró CSV original en la carpeta.")
        df = None
    
    # Inicializar VideoWriter
    OUTPUT_VIDEO_PATH = os.path.join(output_folder, OUTPUT_FILENAME)
    # # out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, FOURCC, 25.0, (WINDOW_WIDTH, WINDOW_HEIGHT))
    out = None # Video desactivado
    out = None # Video desactivado

    seg = PupilSegmenter()
    last_good_roi_mask = initial_roi_bootstrap(image_files, input_folder, seg, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    frame_buffer = collections.deque(maxlen=2)
    current_img_index = 0
    frame_idx_global = 0 
    load_next_frame = True 
    draw_crosshairs = True 
    last_segmentation_results = None
    processed_pair = False
    processed_pair = False
    force_f1_processing = True if last_good_roi_mask is not None else False 
    last_valid_pupil_center = None # Memoria para la posición de la pupila 
    processed_filenames_set = set() # Registro para evitar re-procesar/sobre-escribir frames ya guardados
    
    window_name = "Deteccion Manual (Tecla 'N' para siguiente)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Inicializar Editor Manual
    editor = ManualEditor(window_name)
    cv2.setMouseCallback(window_name, editor.mouse_callback)

    print("----------------------------------------------------------------")
    print(" MODO MANUAL ACTIVO: [N]->Next, [D]->Cruces, [A]->Auto Mode (Toggle), [Q]->Salir")
    print("----------------------------------------------------------------")
    
    csv_data = [] # Inicializar lista para datos CSV

    while True:
        if load_next_frame:
            if current_img_index >= total_images:
                if loop_video:
                    current_img_index = 0
                    frame_buffer.clear()
                    frame_idx_global = 0 
                    force_f1_processing = True if last_good_roi_mask is not None else False 
                else:
                    break
            
            processed_pair = False # Resetear flag al inicio de cada iteración de carga
            
            # --- Carga y Clasificación del Frame ---
            filename = image_files[current_img_index]
            path = os.path.join(input_folder, filename)
            frame = cv2.imread(path)
            if frame is None:
                current_img_index += 1
                continue

            current_img_index += 1
            frame_idx_global += 1
            
            if frame.shape[1] != WINDOW_WIDTH or frame.shape[0] != WINDOW_HEIGHT:
                frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tipo, metrics = PupilTypeClassifier.classify_by_bright_pixels(frame_gray, MIN_BRIGHT_PIXEL_VALUE, MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL)
            
            if not AUTO_MODE:
                print(f"[DEBUG] Frame {frame_idx_global}: Contagem de pixels brilhantes (>{MIN_BRIGHT_PIXEL_VALUE}): {metrics.get('bright_pixel_count', 0)}")
                print(f"Frame {frame_idx_global} classificado como {tipo}")
            
            frame_info = {'frame_num': frame_idx_global, 'frame_data': frame, 'gray_data': frame_gray, 'tipo': tipo, 'metrics': metrics, 'filename': filename}
            frame_buffer.append(frame_info)
            
            current_frame_to_process = frame_buffer[-1]
            f_num = current_frame_to_process['frame_num']
            tipo_f2 = current_frame_to_process['tipo']
            
            # --- Lógica de Segmentación ---
            if f_num == 1 and force_f1_processing and last_good_roi_mask is not None:
                tipo_target = tipo_f2
                frame_target = current_frame_to_process['frame_data']
                last_segmentation_results = seg.segment_pair(frame_target, frame_target, par_idx=f_num, target_type=tipo_target, fallback_roi_mask=last_good_roi_mask)
                force_f1_processing = False 
            elif len(frame_buffer) == 2:
                f2 = frame_buffer[-1]
                f1 = frame_buffer[0]
                tipo_f1 = f1['tipo']
                
                if tipo_f1 != tipo_f2 and tipo_f1 != 'desconhecido' and tipo_f2 != 'desconhecido':
                    frame_escura = f1['frame_data'] if tipo_f1 == 'escura' else f2['frame_data']
                    frame_brilhante = f1['frame_data'] if tipo_f1 == 'brilhante' else f2['frame_data']
                    last_segmentation_results = seg.segment_pair(frame_escura, frame_brilhante, par_idx=f_num, target_type=None, fallback_roi_mask=None)
                    processed_pair = True
                    
                    if last_segmentation_results and np.any(last_segmentation_results['roi_mask']):
                        pupilas_e = last_segmentation_results.get('pupilas_escuras', [])
                        pupilas_b = last_segmentation_results.get('pupilas_brillantes', [])
                        if pupilas_e or pupilas_b:
                            last_good_roi_mask = last_segmentation_results['roi_mask']
                
                elif last_good_roi_mask is not None:
                    frame_target = f2['frame_data']
                    tipo_target = tipo_f2
                    last_segmentation_results = seg.segment_pair(frame_target, frame_target, par_idx=f_num, target_type=tipo_target, fallback_roi_mask=last_good_roi_mask)
                else:
                    last_segmentation_results = None
            else:
                last_segmentation_results = None
                
            # --- PROCESAMIENTO RETROACTIVO DEL FRAME ANTERIOR (SI FUE PAR) ---
            if processed_pair and last_segmentation_results is not None:
                # Recuperar F1 (el frame anterior en el buffer)
                f1_info = frame_buffer[0]
                f1_filename = f1_info['filename']
                f1_tipo = f1_info['tipo']
                
                # Preparar Crosshairs para F1
                f1_crosshair_img = f1_info['frame_data'].copy()
                f1_prefix = 'brilhante' if f1_tipo == 'brilhante' else 'escura'
                f1_pupil_key = 'pupilas_brillantes' if f1_tipo == 'brilhante' else 'pupilas_escuras'
                
                f1_pupilas = last_segmentation_results.get(f1_pupil_key, [])
                f1_glints_raw = last_segmentation_results.get(f'glints_{f1_prefix}_frame', [])
                f1_pupil_mask = last_segmentation_results.get(f'pupil_mask_{f1_prefix}')
                
                # Verificar status de F1 (similar a la lógica principal)
                f1_status = 0
                if not last_segmentation_results.get('blink_detected', False):
                    if f1_pupilas and f1_pupilas[0].get('fit_circle'):
                        f1_status = 1
                        cx, cy, r = f1_pupilas[0]['fit_circle']
                        px, py = cx, cy
                        
                        # Dibujar Pupila
                        draw_cross_full(f1_crosshair_img, (px, py), COLOR_PUPILA, radius=PUPIL_CENTER_RADIUS)
                        
                        # Dibujar Glints
                        if f1_tipo == 'brilhante':
                             g_vis = GlintSelector.select_glints_bright((px, py), f1_glints_raw, f1_pupil_mask)
                             if len(g_vis) >= 1:
                                 draw_cross_full(f1_crosshair_img, g_vis[0]['center'], COLOR_GLINT_BRIGHT, radius=2)
                        else:
                             g_vis = GlintSelector.select_glints_dark((px, py), f1_glints_raw, f1_info['frame_num'])
                             if len(g_vis) >= 1:
                                 draw_cross_full(f1_crosshair_img, g_vis[0]['center'], COLOR_GLINT_1_DARK, radius=2)
                             if len(g_vis) >= 2:
                                 draw_cross_full(f1_crosshair_img, g_vis[1]['center'], COLOR_GLINT_2_DARK, radius=2)
                
                # --- ACTUALIZAR CSV RETROACTIVO PARA F1 ---
                f1_px, f1_py = 0.0, 0.0
                f1_gx, f1_gy = 0.0, 0.0
                f1_vx, f1_vy = 0.0, 0.0
                
                if f1_status == 1:
                    f1_px, f1_py = px, py
                    last_valid_pupil_center = (f1_px, f1_py)
                    
                    # Glints para F1
                    if f1_tipo == 'brilhante':
                        g_coords = GlintSelector.select_glints_bright((f1_px, f1_py), f1_glints_raw, f1_pupil_mask)
                        if len(g_coords) >= 1:
                            f1_gx, f1_gy = g_coords[0]['center']
                    else:
                        g_coords = GlintSelector.select_glints_dark((f1_px, f1_py), f1_glints_raw, f1_info['frame_num'])
                        if len(g_coords) >= 2:
                            f1_gx, f1_gy = _calculate_midpoint(g_coords[0]['center'], g_coords[1]['center'])
                        elif len(g_coords) == 1:
                            f1_gx, f1_gy = g_coords[0]['center']
                            
                elif last_valid_pupil_center is not None and not last_segmentation_results.get('blink_detected', False):
                    # Inferencia por memoria (Solo si NO es parpadeo)
                    f1_px, f1_py = last_valid_pupil_center
                    f1_status = 1
                    
                    # Intentar glints con centro de memoria
                    if f1_tipo == 'brilhante':
                        g_coords = GlintSelector.select_glints_bright((f1_px, f1_py), f1_glints_raw, f1_pupil_mask)
                        if len(g_coords) >= 1:
                            f1_gx, f1_gy = g_coords[0]['center']
                    else:
                        g_coords = GlintSelector.select_glints_dark((f1_px, f1_py), f1_glints_raw, f1_info['frame_num'])
                        if len(g_coords) >= 2:
                            f1_gx, f1_gy = _calculate_midpoint(g_coords[0]['center'], g_coords[1]['center'])
                        elif len(g_coords) == 1:
                            f1_gx, f1_gy = g_coords[0]['center']
                
                if f1_status == 1:
                    if f1_gx != 0.0 and f1_gy != 0.0:
                        f1_vx = f1_px - f1_gx
                        f1_vy = f1_py - f1_gy
                        
                    # Guardar en DataFrame
                    if df is not None and f1_filename not in processed_filenames_set:
                        mask_f1 = df['file_name'] == f1_filename
                        if mask_f1.any():
                            SCALE_X = 1.0/3.0
                            SCALE_Y = 1.0/2.0
                            df.loc[mask_f1, 'status'] = 1
                            df.loc[mask_f1, 'pupil_x'] = float(f"{f1_px * SCALE_X:.2f}")
                            df.loc[mask_f1, 'pupil_y'] = float(f"{f1_py * SCALE_Y:.2f}")
                            df.loc[mask_f1, 'glint_x'] = float(f"{f1_gx * SCALE_X:.2f}")
                            df.loc[mask_f1, 'glint_y'] = float(f"{f1_gy * SCALE_Y:.2f}")
                            df.loc[mask_f1, 'vector_x'] = float(f"{f1_vx * SCALE_X:.2f}")
                            df.loc[mask_f1, 'vector_y'] = float(f"{f1_vy * SCALE_Y:.2f}")
                            processed_filenames_set.add(f1_filename)
                            if not AUTO_MODE:
                                print(f"[RETRO] Actualizado {f1_filename} | Glint: ({f1_gx:.1f},{f1_gy:.1f})")
                
                # (Limpieza de buffer movida al final del ciclo para evitar IndexError en visualización)

            # --- CSV Data Extraction and Logging ---
            current_file_name = frame_info['filename']
            pupil_x, pupil_y = 0.0, 0.0
            glint_x, glint_y = 0.0, 0.0
            vector_x, vector_y = 0.0, 0.0
            status = 0
            
            # Preparar imagen para crosshairs (para guardar)
            crosshair_frame_for_save = frame_info['frame_data'].copy()
            
            if last_segmentation_results is not None:
                prefix = 'brilhante' if tipo_f2 == 'brilhante' else 'escura'
                pupil_key = 'pupilas_brillantes' if tipo_f2 == 'brilhante' else 'pupilas_escuras'
                is_blink_frame = last_segmentation_results.get('blink_detected', False)
                
                pupilas = last_segmentation_results.get(pupil_key, [])
                glints_detected_raw = last_segmentation_results.get(f'glints_{prefix}_frame', [])
                pupil_mask = last_segmentation_results.get(f'pupil_mask_{prefix}')

                if not is_blink_frame:
                    status = 1
                    
                    if pupilas and pupilas[0].get('fit_circle'):
                        cx, cy, r = pupilas[0]['fit_circle']
                        # Coordenadas de la pupila
                        #pupil_x, pupil_y = cx, cy 
                        pupil_x, pupil_y = cx, cy
                        
                        last_valid_pupil_center = (pupil_x, pupil_y) # Actualizar memoria
                        
                        # Recalcular Glints Seleccionados para Coordenadas
                        pupil_center_for_glints = (pupil_x, pupil_y)
                        if tipo_f2 == 'brilhante':
                            # 1 glint: usar su centro
                            glints_for_coords = GlintSelector.select_glints_bright(pupil_center_for_glints, glints_detected_raw, pupil_mask)
                            if len(glints_for_coords) >= 1:
                                glint_x, glint_y = glints_for_coords[0]['center']
                                
                        elif tipo_f2 == 'escura':
                            # 2 glints: usar el punto medio. Si hay 1: usar ese.
                            glints_for_coords = GlintSelector.select_glints_dark(pupil_center_for_glints, glints_detected_raw, f_num)
                            if len(glints_for_coords) >= 2:
                                p1 = glints_for_coords[0]['center']
                                p2 = glints_for_coords[1]['center']
                                glint_x, glint_y = _calculate_midpoint(p1, p2)
                            elif len(glints_for_coords) == 1:
                                glint_x, glint_y = glints_for_coords[0]['center']
                        
                        # Calcular Vector
                        if glint_x != 0.0 and glint_y != 0.0:
                            vector_x = pupil_x - glint_x
                            vector_y = pupil_y - glint_y

                        # --- DIBUJAR CROSSHAIRS PARA GUARDAR ---
                        # Dibujar Pupila (Azul)
                        draw_cross_full(crosshair_frame_for_save, (pupil_x, pupil_y), COLOR_PUPILA, radius=PUPIL_CENTER_RADIUS)
                        
                        # Dibujar Glints
                        if tipo_f2 == 'brilhante':
                             glints_vis = GlintSelector.select_glints_bright((pupil_x, pupil_y), glints_detected_raw, pupil_mask)
                             if len(glints_vis) >= 1:
                                 draw_cross_full(crosshair_frame_for_save, glints_vis[0]['center'], COLOR_GLINT_BRIGHT, radius=2)
                        else:
                             glints_vis = GlintSelector.select_glints_dark((pupil_x, pupil_y), glints_detected_raw, f_num)
                             if len(glints_vis) >= 1:
                                 draw_cross_full(crosshair_frame_for_save, glints_vis[0]['center'], COLOR_GLINT_1_DARK, radius=2)
                             if len(glints_vis) >= 2:
                                 draw_cross_full(crosshair_frame_for_save, glints_vis[1]['center'], COLOR_GLINT_2_DARK, radius=2)

                    # Si el estado es 1, pero las coordenadas de la pupila son 0 (e.g. fallo en el fit), forzar status 0
                    if status == 1 and (pupil_x == 0.0 and pupil_y == 0.0):
                        status = 0 
                        
                # Si el estado es 0, resetear todas las coordenadas a 0.0 (como se solicitó)
                if status == 0:
                    pupil_x, pupil_y = 0.0, 0.0
                    glint_x, glint_y = 0.0, 0.0
                    vector_x, vector_y = 0.0, 0.0
                    
                    # Lógica de Memoria: Si falló (status 0) pero tenemos memoria, pintar la cruz
                    # SOLO si no se detectó parpadeo (respetando la indicación del usuario)
                    if last_valid_pupil_center is not None and not is_blink_frame:
                         draw_cross_full(crosshair_frame_for_save, last_valid_pupil_center, COLOR_PUPILA, radius=PUPIL_CENTER_RADIUS)
                         # save_debug_images(output_folder, current_file_name, last_segmentation_results, tipo_f2, crosshair_frame_for_save)
                         
                         # --- ACTUALIZAR DATOS PARA CSV ---
                         pupil_x, pupil_y = last_valid_pupil_center
                         status = 1 # Marcar como válido (inferred)
                         
                         # Intentar detectar glints usando la posición de memoria
                         if tipo_f2 == 'brilhante':
                              g_coords = GlintSelector.select_glints_bright((pupil_x, pupil_y), glints_detected_raw, pupil_mask)
                              if len(g_coords) >= 1:
                                  glint_x, glint_y = g_coords[0]['center']
                         else:
                              g_coords = GlintSelector.select_glints_dark((pupil_x, pupil_y), glints_detected_raw, f_num)
                              if len(g_coords) >= 2:
                                  glint_x, glint_y = _calculate_midpoint(g_coords[0]['center'], g_coords[1]['center'])
                              elif len(g_coords) == 1:
                                  glint_x, glint_y = g_coords[0]['center']
                         
                         if glint_x != 0.0 and glint_y != 0.0:
                              vector_x = pupil_x - glint_x
                              vector_y = pupil_y - glint_y
                              
                              # Dibujar glints de memoria
                              if tipo_f2 == 'brilhante':
                                   draw_cross_full(crosshair_frame_for_save, (glint_x, glint_y), COLOR_GLINT_BRIGHT, radius=2)
                              else:
                                   draw_cross_full(crosshair_frame_for_save, (glint_x, glint_y), COLOR_GLINT_1_DARK, radius=2)

                else:
                    # Guardar imágenes de depuración SOLO si status=1 (o si se desea ver fallos, pero el usuario pidió estricto)
                    # El usuario pidió: "Debug images and crosshairs should only be saved/drawn if the frame is valid (status = 1)."
                    # AHORA TAMBIEN SI SE USO MEMORIA (arriba)
                    pass
                    # save_debug_images(output_folder, current_file_name, last_segmentation_results, tipo_f2, crosshair_frame_for_save)

            # --- ESCALADO DE COORDENADAS (Interno -> 640x480) ---
            # Factor de escala: X/3, Y/2
            SCALE_X = 1.0/3.0
            SCALE_Y = 1.0/2.0
            
            pupil_x_scaled = pupil_x * SCALE_X
            pupil_y_scaled = pupil_y * SCALE_Y
            glint_x_scaled = glint_x * SCALE_X
            glint_y_scaled = glint_y * SCALE_Y
            vector_x_scaled = vector_x * SCALE_X
            vector_y_scaled = vector_y * SCALE_Y

            # Actualizar DataFrame
            if df is not None and current_file_name not in processed_filenames_set:
                # Buscar fila por file_name
                mask = df['file_name'] == current_file_name
                if mask.any():
                    df.loc[mask, 'status'] = status
                    df.loc[mask, 'pupil_x'] = float(f"{pupil_x_scaled:.2f}")
                    df.loc[mask, 'pupil_y'] = float(f"{pupil_y_scaled:.2f}")
                    df.loc[mask, 'glint_x'] = float(f"{glint_x_scaled:.2f}")
                    df.loc[mask, 'glint_y'] = float(f"{glint_y_scaled:.2f}")
                    df.loc[mask, 'vector_x'] = float(f"{vector_x_scaled:.2f}")
                    df.loc[mask, 'vector_y'] = float(f"{vector_y_scaled:.2f}")
                    processed_filenames_set.add(current_file_name)
                else:
                    print(f"[WARN] No se encontró {current_file_name} en el CSV original")
            
            load_next_frame = False
            
        # --- Visualización ---
        f_display = frame_buffer[-1]
        display_frame = f_display['frame_data'].copy()
        displayed_frame_num = f_display['frame_num']
        tipo_f2 = f_display['tipo']
        bright_pixel_count = f_display.get('metrics', {}).get('bright_pixel_count', 0)
        detection_status = f"Img: {displayed_frame_num}/{total_images} | TIPO: {tipo_f2.upper()}"

        # Recalcular pupil_center y glints_detected_filtered para la visualización
        pupil_center_vis = None
        glints_detected_filtered = []
        is_blink_frame = False 
        
        # Inicializar variables de visualización por defecto
        h_sub = WINDOW_HEIGHT // 2
        w_sub = WINDOW_WIDTH // 3
        sub_frame_roi_mask_bgr = np.zeros((h_sub, w_sub, 3), dtype=np.uint8)
        sub_frame_diff_bgr = np.zeros((h_sub, w_sub, 3), dtype=np.uint8)
        sub_frame_diff_thresh_bgr = np.zeros((h_sub, w_sub, 3), dtype=np.uint8)
        sub_frame_bitwise_and_bgr = np.zeros((h_sub, w_sub, 3), dtype=np.uint8)
        sub_frame_glints_thresh_bgr = np.zeros((h_sub, w_sub, 3), dtype=np.uint8)
        roi_tag = "ROI (Diff | Par)"
        diff_tag = "Diff Raw (N/A)"
        diff_thresh_tag = "Diff Thresh (N/A)"
        bitwise_tag = "Bitwise (N/A)"
        thresh_tag = "Thresh (N/A)"
 
        
        if last_segmentation_results is not None:
            prefix = 'brilhante' if tipo_f2 == 'brilhante' else 'escura'
            pupil_key = 'pupilas_brillantes' if tipo_f2 == 'brilhante' else 'pupilas_escuras'
            is_blink_frame = last_segmentation_results.get('blink_detected', False)

            pupilas = last_segmentation_results.get(pupil_key, [])
            pupil_mask = last_segmentation_results.get(f'pupil_mask_{prefix}')

            if pupilas and pupilas[0].get('fit_circle'):
                cx, cy, r = pupilas[0]['fit_circle']
                pupil_center_vis = (int(cx), int(cy))
                detection_status += " | Seg OK"
                is_blink_frame = False 
            else:
                detection_status += " | BLINK DETECTADO"
                is_blink_frame = last_segmentation_results.get('blink_detected', True) # Usar el resultado final

            glints_detected_raw = last_segmentation_results.get(f'glints_{prefix}_frame', [])
            if pupil_center_vis:
                if tipo_f2 == 'brilhante':
                    glints_detected_filtered = GlintSelector.select_glints_bright(pupil_center_vis, glints_detected_raw, pupil_mask)
                else:
                    glints_detected_filtered = GlintSelector.select_glints_dark(pupil_center_vis, glints_detected_raw, f_num)
            else:
                # Caso fallback: NO hay pupila, pero SÍ pueden haber glints
                # Usar TODOS los glints detectados sin filtrar
                glints_detected_filtered = glints_detected_raw if glints_detected_raw else []
            
            # ... (Resto de la lógica de visualización de mosaico) ...
            # ... (Resto de la lógica de visualización de mosaico) ...

            roi_mask_to_show = last_segmentation_results.get('roi_mask')
            if roi_mask_to_show is not None and np.any(roi_mask_to_show):
                sub_frame_roi_mask_bgr = cv2.resize(cv2.cvtColor(roi_mask_to_show, cv2.COLOR_GRAY2BGR), (w_sub, h_sub))
                
                is_fallback_roi = (last_good_roi_mask is not None and np.array_equal(roi_mask_to_show, last_good_roi_mask))
                if is_fallback_roi and (len(frame_buffer) < 2 or (len(frame_buffer) == 2 and f1['tipo'] == tipo_f2)):
                     roi_tag = "ROI (FALLBACK)"
                elif f_num == 1 and force_f1_processing is False:
                     roi_tag = "ROI (F1 FORZADO)"
                else: 
                     roi_tag = "ROI (Diff | Par)"

            # Fitted Circle Visualization (Bottom Right)
            # Create image showing the eye image with the fitted circle from watershed
            if tipo_f2 == 'brilhante':
                base_img = last_segmentation_results.get('brilhante_gray')
            else:
                base_img = last_segmentation_results.get('escura_gray')
            
            if base_img is not None:
                circle_vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
                if pupilas and pupilas[0].get('fit_circle'):
                    cx, cy, r = pupilas[0]['fit_circle']
                    # Draw the fitted circle in green
                    cv2.circle(circle_vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                    # Draw center point in red
                    cv2.circle(circle_vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    diff_thresh_tag = f"Fitted Circle (R={int(r)})"
                else:
                    diff_thresh_tag = "Fitted Circle (N/A)"
                sub_frame_diff_thresh_bgr = cv2.resize(circle_vis, (w_sub, h_sub))
            else:
                sub_frame_diff_thresh_bgr = np.zeros((h_sub, w_sub, 3), dtype=np.uint8)
                diff_thresh_tag = "Fitted Circle (No Image)"

            diff_img_to_show = last_segmentation_results.get('diferenca')
            if diff_img_to_show is not None:
                sub_frame_diff_bgr = cv2.resize(cv2.cvtColor(diff_img_to_show, cv2.COLOR_GRAY2BGR), (w_sub, h_sub))
                diff_tag = "Diff Raw (Subtract)"

            bitwise_img_to_show = last_segmentation_results.get(f'{prefix}_bitwise_and')
            if bitwise_img_to_show is not None:
                img_gray = bitwise_img_to_show
                sub_frame_bitwise_and_bgr = cv2.resize(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), (w_sub, h_sub))
                bitwise_tag = f"Bitwise ({tipo_f2.upper()} | F{displayed_frame_num})"
                
            thresh_img_to_show = last_segmentation_results.get(f'{prefix}_thresh')
            if thresh_img_to_show is not None:
                img_gray = thresh_img_to_show
                sub_frame_glints_thresh_bgr = cv2.resize(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), (w_sub, h_sub))
                thresh_tag = f"Thresh Glints ({tipo_f2.upper()} | F{displayed_frame_num})"
        else:
            detection_status += " | Esperando proceso..."
            pupil_center_vis = None
            glints_detected_filtered = []

        if draw_crosshairs:
            # Dibujar pupila - usar detección O memoria
            pos_to_draw = pupil_center_vis if pupil_center_vis else last_valid_pupil_center
            if pos_to_draw:
                draw_cross_full(display_frame, pos_to_draw, COLOR_PUPILA, radius=PUPIL_CENTER_RADIUS + 2)
            
            # Dibujar glints SIEMPRE que estén detectados (independiente de la pupila)
            if glints_detected_filtered:
                if tipo_f2 == 'brilhante':
                    if len(glints_detected_filtered) >= 1:
                        draw_cross_full(display_frame, glints_detected_filtered[0]['center'], COLOR_GLINT_BRIGHT, radius=GLINT_RADIUS_BRIGHT + 2)
                else:  # escura
                    if len(glints_detected_filtered) > 0:
                        draw_cross_full(display_frame, glints_detected_filtered[0]['center'], COLOR_GLINT_1_DARK, radius=GLINT_RADIUS_DARK + 2)
                    if len(glints_detected_filtered) > 1:
                        draw_cross_full(display_frame, glints_detected_filtered[1]['center'], COLOR_GLINT_2_DARK, radius=GLINT_RADIUS_DARK + 2)


        if display_frame is not None:
            status_text = f"FRAME {displayed_frame_num} | Cruces: {'ON' if draw_crosshairs else 'OFF'} | {detection_status}"
            text_color = (0, 0, 255) if is_blink_frame else (255, 255, 255)
            cv2.putText(display_frame, status_text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            if is_blink_frame:
                 cv2.putText(display_frame, "PARPADEO / PUPILA INVALIDA", (WINDOW_WIDTH//2 - 200, WINDOW_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

            bright_count_text = f"Pixeles >= {MIN_BRIGHT_PIXEL_VALUE}: {int(bright_pixel_count)}"
            color_count = COLOR_TEXTO_CONTEO_CRITICO if bright_pixel_count >= MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL else COLOR_TEXTO_CONTEO_NORMAL
            cv2.putText(display_frame, bright_count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_count, 2, cv2.LINE_AA)
            
            # Crear mosaico
            display_frame_resized = cv2.resize(display_frame, (w_sub, h_sub))
            cv2.putText(display_frame_resized, f"Actual F{displayed_frame_num} ({tipo_f2.upper()})", (10, h_sub-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_MOSAICO, 1, cv2.LINE_AA)
            cv2.putText(sub_frame_roi_mask_bgr, roi_tag, (10, h_sub-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_MOSAICO, 1, cv2.LINE_AA)
            cv2.putText(sub_frame_diff_bgr, diff_tag, (10, h_sub-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_MOSAICO, 1, cv2.LINE_AA)
            
            cv2.putText(sub_frame_bitwise_and_bgr, bitwise_tag, (10, h_sub-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_MOSAICO, 1, cv2.LINE_AA)
            cv2.putText(sub_frame_glints_thresh_bgr, thresh_tag, (10, h_sub-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_MOSAICO, 1, cv2.LINE_AA)
            cv2.putText(sub_frame_diff_thresh_bgr, diff_thresh_tag, (10, h_sub-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_MOSAICO, 1, cv2.LINE_AA)

            top_row = np.hstack((display_frame_resized, sub_frame_roi_mask_bgr, sub_frame_diff_bgr))
            bottom_row = np.hstack((sub_frame_bitwise_and_bgr, sub_frame_glints_thresh_bgr, sub_frame_diff_thresh_bgr))
            mosaic_frame = np.vstack((top_row, bottom_row))

            # Escribir al archivo de video
            if out is not None and out.isOpened():
                out.write(mosaic_frame)
                
            if not AUTO_MODE:
                cv2.imshow(window_name, mosaic_frame)
            elif displayed_frame_num % 10 == 0:
                print(f"[i] Procesando Frame {displayed_frame_num}/{total_images} (Auto Mode - No GUI)...")

        # --- Control de Teclado ---
        #    AUTO_MODE = False  # Set to True for automatic processing, no esperamos tecla (ni mostramos ventana), avanzamos directo.
        if not AUTO_MODE:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("[i] Salida solicitada por usuario.")
                break
            elif key == ord('d'): # MODO EDICIÓN
                # RESETEAR todas las variables para evitar valores residuales
                pupil_x, pupil_y = 0.0, 0.0
                glint_x, glint_y = 0.0, 0.0
                vector_x, vector_y = 0.0, 0.0
                status = 0
                
                # Iniciar edición
                editor.start_editing(tipo_f2, w_sub, h_sub)
                
                while True:
                    # Mostrar estado de edición
                    edit_frame = mosaic_frame.copy()
                    cv2.putText(edit_frame, f"EDITANDO: Click {len(editor.clicks)}/{2 if tipo_f2=='brilhante' else 3}", (10, WINDOW_HEIGHT-60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(edit_frame, "P: Parpadeo | Esc: Cancelar", (10, WINDOW_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Dibujar clicks actuales
                    for pt in editor.clicks:
                        # Dibujar en el cuadrante superior izquierdo (Mapeo inverso: X/3, Y/2)
                        pt_vis = (int(pt[0]/3), int(pt[1]/2))
                        cv2.circle(edit_frame, pt_vis, 3, (0, 0, 255), -1)
                        
                    cv2.imshow(window_name, edit_frame)
                    k = cv2.waitKey(10) & 0xFF
                    
                    if k == 27: # Esc
                        editor.active = False
                        print("[EDIT] Cancelado")
                        break
                        
                    if k == ord('p') or k == ord('P'): # Parpadeo
                        status = 0
                        pupil_x, pupil_y = 0.0, 0.0
                        glint_x, glint_y = 0.0, 0.0
                        vector_x, vector_y = 0.0, 0.0
                        
                        # Actualizar CSV (escalado 0.5 ya incluido en update_csv_row si pasamos valores raw? NO, update espera valores finales)
                        # Pero espera, update_csv_row escribe directo al DF.
                        # Las coordenadas en el DF deben estar ESCALADAS (0.5).
                        # Si status=0, todo es 0.0
                        update_csv_row(df, current_file_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        
                        editor.active = False
                        print("[EDIT] Marcado como PARPADEO")
                        # Actualizar visualización principal para reflejar cambio
                        detection_status = "EDITADO: PARPADEO"
                        is_blink_frame = True
                        draw_crosshairs = False # Ocultar cruces
                        break
                        
                    if editor.done:
                        # Procesar clicks
                        clicks = editor.clicks
                        status = 1
                        
                        # Pupila (Click 1)
                        pupil_x, pupil_y = clicks[0]
                        
                        # Glints
                        if tipo_f2 == 'brilhante':
                            # Click 2: Glint
                            glint_x, glint_y = clicks[1]
                        else:
                            # Click 2 y 3: Glints (Punto medio)
                            p1 = clicks[1]
                            p2 = clicks[2]
                            glint_x = (p1[0] + p2[0]) / 2
                            glint_y = (p1[1] + p2[1]) / 2
                        
                        # Vector
                        vector_x = pupil_x - glint_x
                        vector_y = pupil_y - glint_y
                        
                        # ESCALAR para CSV (X:1/3, Y:1/2)
                        SCALE_X = 1.0/3.0
                        SCALE_Y = 1.0/2.0
                        px_s, py_s = pupil_x * SCALE_X, pupil_y * SCALE_Y
                        gx_s, gy_s = glint_x * SCALE_X, glint_y * SCALE_Y
                        vx_s, vy_s = vector_x * SCALE_X, vector_y * SCALE_Y
                        
                        update_csv_row(df, current_file_name, 1, px_s, py_s, gx_s, gy_s, vx_s, vy_s)
                        processed_filenames_set.add(current_file_name)
                        
                        editor.active = False
                        print(f"[EDIT] Guardado: P({pupil_x},{pupil_y}) G({glint_x:.1f},{glint_y:.1f})")
                        
                        # Actualizar visualización principal
                        pupil_center_vis = (int(pupil_x), int(pupil_y))
                        # Glints para visualización (hack: crear lista de dicts)
                        glints_detected_filtered = []
                        if tipo_f2 == 'brilhante':
                            glints_detected_filtered.append({'center': (int(glint_x), int(glint_y))})
                        else:
                            # Reconstruir glints individuales para visualización
                            glints_detected_filtered.append({'center': clicks[1]})
                            glints_detected_filtered.append({'center': clicks[2]})
                        
                        detection_status = "EDITADO MANUALMENTE"
                        is_blink_frame = False
                        draw_crosshairs = True
                        break
            elif key == ord('n'):
                load_next_frame = True
            elif key == ord('c'):
                draw_crosshairs = not draw_crosshairs
                print(f"[i] Dibujar Crosshairs: {draw_crosshairs}")
            elif key == ord('a'):
                AUTO_MODE = not AUTO_MODE
                state_str = "ACTIVADO" if AUTO_MODE else "DESACTIVADO"
                print(f"[i] MODO AUTOMATICO {state_str}")
                if AUTO_MODE:
                    print("[i] Visualización desactivada para mayor velocidad. Presione Ctrl+C para detener.")
                    load_next_frame = True 
        else:
            # En modo automático, forzamos carga del siguiente frame
            load_next_frame = True

        # (Limpieza de buffer revertida para recuperar fluidez de visualización)
            
        # (Eliminado incremento redundante de current_img_index)
            
    # Liberar recursos
    if out is not None and out.isOpened():
        out.release()
    cv2.destroyAllWindows()

    # --- Lógica de Escritura de CSV ---
    if df is not None:
        # Reordenar columnas para poner vector_x y vector_y después de tar6_y si existe
        cols = list(df.columns)
        if 'tar6_y' in cols and 'vector_x' in cols and 'vector_y' in cols:
            # Remover vector_x y vector_y de su posición actual
            cols = [c for c in cols if c not in ['vector_x', 'vector_y']]
            # Insertar después de tar6_y
            idx = cols.index('tar6_y') + 1
            cols.insert(idx, 'vector_x')
            cols.insert(idx + 1, 'vector_y')
            df = df[cols]
        
        # Nombre del CSV procesado
        base_name = os.path.basename(original_csv_path)
        name_parts = os.path.splitext(base_name)
        processed_name = f"{name_parts[0]}_processed{name_parts[1]}"
        CSV_FULL_PATH = os.path.join(output_folder, processed_name)
        
        try:
            df.to_csv(CSV_FULL_PATH, index=False)
            print(f"[i] CSV procesado guardado en: {CSV_FULL_PATH}")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar el archivo CSV: {e}")
    else:
        print("[ERROR] No hay DataFrame para guardar")
    # --- Fin Lógica de Escritura de CSV ---

    print("\n[i] Terminado.")

def parse_args():
    p = argparse.ArgumentParser(description="Procesamiento manual paso a paso de secuencia de imágenes de ojo.")
    p.add_argument("--input_folder", "-i", required=False, 
                   #default=r"data/P01/fp1",
                   #default =r"data/P19/fp1",
                   default =r"data/P10/fp1")
                   #help="Ruta a la carpeta con la secuencia de imágenes.")
    p.add_argument("--loop", action="store_true", help="Reproducir secuencia en bucle.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input_folder, loop_video=args.loop)

