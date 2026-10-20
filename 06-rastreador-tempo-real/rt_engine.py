"""
rt_engine_v2.py
===============
Motor de calibração e estimação de olhar.

Responsabilidades:
  - Receber vetores pupila-glint do detector.
  - Treinar modelo de regressão polinomial Ridge durante calibração.
  - Converter vetores em coordenadas de tela (predição de olhar).
  - Gerar relatório final com CSV, estatísticas e heatmap.
"""

import multiprocessing as mp
import time
import csv
import queue as _queue


class RT_Engine_v2(mp.Process):

    def __init__(self, vector_queue, gaze_queue, command_queue, config,
                 model_ready_flag=None):
        """
        model_ready_flag: mp.Value('b', False) compartilhado com o orquestrador.
        Definido como True quando o modelo está treinado e pronto.
        """
        super().__init__(daemon=True)
        self.vector_queue    = vector_queue
        self.gaze_queue      = gaze_queue
        self.command_queue   = command_queue
        self.config          = config
        self.model_ready_flag = model_ready_flag
        self.running         = mp.Value('b', True)

        self.state            = 'IDLE'
        self.current_target   = (960, 540)
        self.current_seq_info = ('None', 0)

        self.calib_buffer   = {}
        self.model_x        = None
        self.model_y        = None
        self.estimation_log = []

        # Limites de tela para clampear predições
        self.SCREEN_W = config.get('screen_width',  1920)
        self.SCREEN_H = config.get('screen_height', 1080)

        # Física do setup
        self.PIXEL_PITCH_MM = (config.get('monitor_width_mm',  507.0) /
                               config.get('monitor_width_px', 1920))
        self.DISTANCE_MM    = config.get('distance_mm', 600.0)

        self.VECS_PER_POINT = config.get('calib_vectors_per_point', 999)  # sin límite máximo
        self.POLY_DEGREE    = config.get('poly_degree', 2)
        self.RIDGE_ALPHA    = config.get('ridge_alpha', 5.0)

        # Limiar para descartar vetores outlier
        self.MAX_VECTOR_PX  = config.get('max_vector_px', 35)

    def px_to_deg(self, delta_px):
        import numpy as np
        mm = abs(delta_px) * self.PIXEL_PITCH_MM
        return float(np.degrees(np.arctan(mm / self.DISTANCE_MM)))

    def error_deg(self, pred_x, pred_y, tar_x, tar_y):
        import numpy as np
        dp = np.sqrt((pred_x - tar_x)**2 + (pred_y - tar_y)**2)
        return self.px_to_deg(dp)

    def _train(self):
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        X_list, tx_list, ty_list = [], [], []
        for (tx, ty), vecs in self.calib_buffer.items():
            if len(vecs) == 0:
                print(f"RT_Engine: Punto ({tx},{ty}) sin vectores, ignorado.")
                continue
            arr = np.array(vecs)
            if len(arr) >= 3:
                mu, sig = arr.mean(0), arr.std(0) + 1e-6
                mask  = np.all(np.abs(arr - mu) < 2.5 * sig, axis=1)
                clean = arr[mask] if mask.sum() >= 1 else arr
            else:
                clean = arr
            X_list.append(clean.mean(0))
            tx_list.append(tx); ty_list.append(ty)
            print(f"RT_Engine: Punto ({tx},{ty}) -> {len(vecs)} vecs -> media {clean.mean(0).round(3)}")

        n = len(X_list)
        print(f"RT_Engine: Entrenando con {n} puntos de calibracion.")
        if n < 3:
            print("RT_Engine: ERROR — se necesitan al menos 3 puntos validos.")
            return False

        X  = np.array(X_list)
        tx = np.array(tx_list, dtype=float)
        ty = np.array(ty_list, dtype=float)

        self.model_x = make_pipeline(
            PolynomialFeatures(degree=self.POLY_DEGREE, include_bias=True),
            Ridge(alpha=self.RIDGE_ALPHA))
        self.model_y = make_pipeline(
            PolynomialFeatures(degree=self.POLY_DEGREE, include_bias=True),
            Ridge(alpha=self.RIDGE_ALPHA))
        self.model_x.fit(X, tx)
        self.model_y.fit(X, ty)

        # Validación LOO
        from sklearn.model_selection import LeaveOneOut
        errs = []
        for tr, te in LeaveOneOut().split(X):
            mx = make_pipeline(PolynomialFeatures(self.POLY_DEGREE, include_bias=True),
                               Ridge(alpha=self.RIDGE_ALPHA))
            my = make_pipeline(PolynomialFeatures(self.POLY_DEGREE, include_bias=True),
                               Ridge(alpha=self.RIDGE_ALPHA))
            mx.fit(X[tr], tx[tr]); my.fit(X[tr], ty[tr])
            px_ = float(np.clip(mx.predict(X[te])[0], 0, self.SCREEN_W))
            py_ = float(np.clip(my.predict(X[te])[0], 0, self.SCREEN_H))
            errs.append(self.error_deg(px_, py_, tx[te][0], ty[te][0]))

        print(f"RT_Engine: Calibração OK. LOO erro = {np.mean(errs):.2f}° ± {np.std(errs):.2f}°")
        return True

    def run(self):
        import numpy as np
        print("RT_Engine: Iniciado.")
        while self.running.value:
            # Comandos
            while not self.command_queue.empty():
                try:
                    cmd, params = self.command_queue.get_nowait()
                    self._handle(cmd, params)
                except _queue.Empty:
                    break

            # Vectores
            processed = 0
            while not self.vector_queue.empty() and processed < 20:
                try:
                    data = self.vector_queue.get_nowait()
                    processed += 1
                    self._process(*data)
                except _queue.Empty:
                    break
            time.sleep(0.002)
        print("RT_Engine: Detenido.")

    def _handle(self, cmd, params):
        if cmd == 'START_CALIB':
            self.state        = 'CALIBRATING'
            self.calib_buffer = {}
            if self.model_ready_flag:
                self.model_ready_flag.value = False
            print("RT_Engine: Estado → CALIBRATING")

        elif cmd == 'CALIB_POINT':
            self.current_target = tuple(params)

        elif cmd == 'STOP_CALIB':
            ok = self._train()
            self.state = 'IDLE' if ok else 'ERROR'
            print(f"RT_Engine: Estado → {'IDLE (modelo listo)' if ok else 'ERROR'}")
            # Sinalizar ao orquestrador que o modelo está pronto
            if ok and self.model_ready_flag:
                self.model_ready_flag.value = True
                print("RT_Engine: MODEL_READY → True")

        elif cmd == 'START_SEQUENCE':
            seq_name, tid, x, y = params
            self.state            = 'ESTIMATING'
            self.current_target   = (x, y)
            self.current_seq_info = (seq_name, tid)

        elif cmd == 'IDLE':
            self.state = 'IDLE'

        elif cmd == 'GENERATE_REPORT':
            self._save_report()

    def _process(self, ts, vx, vy, px, py, gx, gy):
        import numpy as np

        # Descartar vetores outlier
        if abs(vx) > self.MAX_VECTOR_PX or abs(vy) > self.MAX_VECTOR_PX:
            return

        vec = np.array([[vx, vy]])

        if self.state == 'CALIBRATING':
            key = self.current_target
            if key not in self.calib_buffer:
                self.calib_buffer[key] = []
            self.calib_buffer[key].append((vx, vy))
            n = len(self.calib_buffer[key])
            if n % 10 == 0:
                print(f'RT_Engine: Calib punto {key} -> {n} vetores acumulados')

        elif self.state == 'ESTIMATING' and self.model_x is not None:
            raw_x = float(self.model_x.predict(vec)[0])
            raw_y = float(self.model_y.predict(vec)[0])

            pred_x = float(np.clip(raw_x, 0, self.SCREEN_W))
            pred_y = float(np.clip(raw_y, 0, self.SCREEN_H))

            err = self.error_deg(pred_x, pred_y,
                                 self.current_target[0], self.current_target[1])
            seq_name, tid = self.current_seq_info

            # Registrar no log de estimação
            self.estimation_log.append({
                'ts':        ts,
                'sequence':  seq_name,
                'target_id': tid,
                'target_x':  self.current_target[0],
                'target_y':  self.current_target[1],
                'pred_x':    round(pred_x, 2),
                'pred_y':    round(pred_y, 2),
                'raw_pred_x': round(raw_x, 2),
                'raw_pred_y': round(raw_y, 2),
                'pupil_x':   round(px, 3),
                'pupil_y':   round(py, 3),
                'glint_x':   round(gx, 3),
                'glint_y':   round(gy, 3),
                'vx':        round(vx, 4),
                'vy':        round(vy, 4),
                'error_deg': round(err, 4),
            })

            if not self.gaze_queue.full():
                self.gaze_queue.put_nowait((ts, pred_x, pred_y))

    def _save_report(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import defaultdict

        if not self.estimation_log:
            print("RT_Engine: No hay datos de estimación para reportar.")
            return

        # CSV completo
        csv_path = 'rt_estimation_results_v2.csv'
        fields   = ['ts', 'sequence', 'target_id', 'target_x', 'target_y',
                    'pred_x', 'pred_y', 'raw_pred_x', 'raw_pred_y',
                    'pupil_x', 'pupil_y', 'glint_x', 'glint_y',
                    'vx', 'vy', 'error_deg']
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(self.estimation_log)
        print(f"RT_Engine: CSV guardado → {csv_path}")

        all_errors = [d['error_deg'] for d in self.estimation_log]
        mean_err   = float(np.mean(all_errors))
        std_err    = float(np.std(all_errors))
        med_err    = float(np.median(all_errors))

        # FIX 5: estadísticas por cuadrante
        def quadrant(tx, ty):
            if tx <  640 and ty <  360: return 'sup-izq'
            if tx > 1280 and ty <  360: return 'sup-der'
            if tx <  640 and ty >  720: return 'inf-izq'
            if tx > 1280 and ty >  720: return 'inf-der'
            return 'centro'

        quad_errors = defaultdict(list)
        for d in self.estimation_log:
            quad_errors[quadrant(d['target_x'], d['target_y'])].append(d['error_deg'])

        # % predicciones dentro de pantalla
        in_screen = sum(1 for d in self.estimation_log
                        if 0 <= d['pred_x'] <= self.SCREEN_W and
                           0 <= d['pred_y'] <= self.SCREEN_H)
        pct_in = 100 * in_screen / len(self.estimation_log)

        print(f"\n{'='*52}")
        print(f"  REPORTE FINAL — RT Eye Tracker v2")
        print(f"  Muestras          : {len(all_errors)}")
        print(f"  Error medio       : {mean_err:.2f}°")
        print(f"  Desv. padrão      : {std_err:.2f}°")
        print(f"  Mediana           : {med_err:.2f}°")
        print(f"  Pred dentro pantalla: {pct_in:.1f}%")
        print(f"  Error por cuadrante:")
        for q, errs in sorted(quad_errors.items()):
            print(f"    {q:10s}: {np.mean(errs):.2f}° ± {np.std(errs):.2f}° (n={len(errs)})")
        print(f"{'='*52}\n")

        # Stats por target
        target_data = defaultdict(list)
        target_pos  = {}
        for d in self.estimation_log:
            tid = d['target_id']
            target_data[tid].append(d['error_deg'])
            target_pos[tid] = (d['target_x'], d['target_y'])

        stats_path = 'rt_stats_per_target_v2.csv'
        with open(stats_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['target_id', 'target_x', 'target_y', 'n_samples',
                        'mean_deg', 'std_deg', 'median_deg', 'min_deg', 'max_deg'])
            for tid in sorted(target_data.keys()):
                errs = target_data[tid]
                tx, ty = target_pos[tid]
                w.writerow([tid, tx, ty, len(errs),
                            round(np.mean(errs), 4), round(np.std(errs), 4),
                            round(np.median(errs), 4), round(np.min(errs), 4),
                            round(np.max(errs), 4)])
        print(f"RT_Engine: Estadísticas por target → {stats_path}")

        # Heatmap viridis 5×7
        grid = np.full((5, 7), np.nan)
        for tid, errs in target_data.items():
            r = (tid-1)//7; c = (tid-1)%7
            if 0 <= r < 5 and 0 <= c < 7:
                grid[r, c] = float(np.mean(errs))
        try:
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(grid, annot=True, fmt='.2f', cmap='viridis',
                        linewidths=0.5, linecolor='gray',
                        cbar_kws={'label': 'Error angular (°)'},
                        ax=ax, mask=np.isnan(grid))
            ax.set_title(f'Mapa de Error Angular — RT Eye Tracker v2\n'
                         f'Media: {mean_err:.2f}°  DP: {std_err:.2f}°  '
                         f'Mediana: {med_err:.2f}°',
                         fontsize=13, fontweight='bold')
            ax.set_xticklabels([f'C{i+1}' for i in range(7)])
            ax.set_yticklabels([f'F{i+1}' for i in range(5)], rotation=0)
            ax.set_xlabel('Columna de target')
            ax.set_ylabel('Fila de target')
            plt.tight_layout()
            plt.savefig('rt_heatmap_v2.png', dpi=150)
            plt.close()
            print("RT_Engine: Heatmap guardado → rt_heatmap_v2.png")
        except Exception as e:
            print(f"RT_Engine: Error generando heatmap: {e}")

        print("RT_Engine: Reporte completo.")

    def stop(self):
        self.running.value = False