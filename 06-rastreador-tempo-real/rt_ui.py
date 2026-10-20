"""
rt_ui_v2.py
===========
Interfaz del RT Eye Tracker v2

"""

import multiprocessing as mp
import time
import queue as _queue
import threading
import random
import math


class PreviewWindows:
    WIN_B = 'Pupila BRILHANTE (frames impares) — ON-AXIS'
    WIN_D = 'Pupila ESCURA    (frames pares)   — OFF-AXIS'

    def __init__(self, preview_queue, stop_event):
        self.preview_queue = preview_queue
        self.stop_event    = stop_event

    def run(self):
        import cv2
        import numpy as np
        cv2.namedWindow(self.WIN_B, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.WIN_D, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN_B, 640, 480)
        cv2.resizeWindow(self.WIN_D, 640, 480)
        cv2.moveWindow(self.WIN_B, 20,  80)
        cv2.moveWindow(self.WIN_D, 680, 80)
        last_b = np.zeros((480,640,3), dtype=np.uint8)
        last_d = np.zeros((480,640,3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while not self.stop_event.is_set():
            latest = None
            while not self.preview_queue.empty():
                try: latest = self.preview_queue.get_nowait()
                except _queue.Empty: break
            if latest is not None:
                ts, fidx, frame = latest
                if frame.shape[:2] != (480,640):
                    frame = cv2.resize(frame, (640,480))
                if fidx % 2 == 1: last_b = frame.copy()
                else:              last_d = frame.copy()
            bd = last_b.copy(); dd = last_d.copy()
            cv2.putText(bd,'ON-AXIS  | glint central',(8,24),font,0.6,(0,255,100),2)
            cv2.putText(bd,'Pupila CLARA — frame impar',(8,50),font,0.5,(0,255,100),1)
            cv2.putText(bd,'Presione ESPACIO para continuar',(8,460),font,0.5,(200,200,200),1)
            cv2.putText(dd,'OFF-AXIS | 2 glints laterales',(8,24),font,0.6,(0,200,255),2)
            cv2.putText(dd,'Pupila OSCURA — frame par',(8,50),font,0.5,(0,200,255),1)
            cv2.putText(dd,'Presione ESPACIO para continuar',(8,460),font,0.5,(200,200,200),1)
            cv2.imshow(self.WIN_B, bd)
            cv2.imshow(self.WIN_D, dd)
            key = cv2.waitKey(16) & 0xFF
            if key in (32, 27): self.stop_event.set(); break
        cv2.destroyAllWindows()


class RT_UI_v2(mp.Process):

    X_COORDS = [135, 410, 685, 960, 1235, 1510, 1785]
    Y_COORDS = [108, 324, 540,  756,  972]
    CALIB_IDS = [1, 4, 7, 15, 18, 21, 29, 32, 35]
    FIX_CENTER_ID = 18
    FIXACOES_GROUPS = {
        'Canto_SE': [9,  8,  1,  2,  3],
        'Canto_SD': [13, 14,  7,  6,  5],
        'Anc11':    [11, 12,  4, 10, 16],
        'Anc23':    [23, 22, 30, 24, 16],
        'Anc27':    [27, 20, 26, 34, 28],
        'W':        [15, 31, 25, 33, 21],
        'M':        [29, 17, 32, 19, 35],
    }

    def __init__(self, command_queue, gaze_queue, preview_queue, config,
                 signal_queue=None):
        super().__init__(daemon=True)
        self.command_queue = command_queue
        self.gaze_queue    = gaze_queue
        self.preview_queue = preview_queue
        self.signal_queue  = signal_queue
        self.config        = config
        self.W = config.get('screen_width',  1920)
        self.H = config.get('screen_height', 1080)
        self.running = mp.Value('b', True)

    def run(self):
        # Fase 1: preview diagnóstico
        stop_ev = threading.Event()
        pw = PreviewWindows(self.preview_queue, stop_ev)
        t  = threading.Thread(target=pw.run, daemon=True)
        t.start()
        print("RT_UI: Mostrando janelas de diagnóstico...")
        print("       Pressione ESPAÇO nas janelas para continuar.")
        t.join()
        while not self.preview_queue.empty():
            try: self.preview_queue.get_nowait()
            except Exception: pass
        print("RT_UI: Preview fechado. Iniciando interface de experimento.")
        self._run_tkinter()

    def _run_tkinter(self):
        import tkinter as tk
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

        root   = tk.Tk()
        root.attributes('-fullscreen', True)
        root.config(bg='black', cursor='none')
        root.update_idletasks()

        self.W = root.winfo_screenwidth()
        self.H = root.winfo_screenheight()
        print(f"RT_UI: Resolución detectada = {self.W}x{self.H}")

        # Recalcular coordenadas 7x5 dinamicamente
        X_PCTS = [0.07, 0.21, 0.35, 0.50, 0.64, 0.78, 0.93]
        Y_PCTS = [0.10, 0.30, 0.50, 0.70, 0.90]
        self.X_COORDS = [int(p * self.W) for p in X_PCTS]
        self.Y_COORDS = [int(p * self.H) for p in Y_PCTS]

        canvas = tk.Canvas(root, width=self.W, height=self.H,
                           bg='black', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        root.bind('<Escape>', lambda e: self._quit(root))

        centers  = [(x, y) for y in self.Y_COORDS for x in self.X_COORDS]
        seq_keys = list(self.FIXACOES_GROUPS.keys())
        state    = {'name': 'START_SCREEN', 'calib_idx': 0,
                    'seq_idx': 0, 'point_in_seq': 0,
                    'model_ready': False, 'simulated': False,
                    'last_target_pos': (self.W/2, self.H/2)}

        def clear():
            canvas.delete('all')

        def draw_text(text, color='white', size=22, y_frac=0.5):
            canvas.create_text(self.W/2, self.H*y_frac, text=text,
                               fill=color, font=('Courier', size, 'bold'),
                               justify=tk.CENTER, tags='info')

        def draw_cross(x, y, size=30, color='white', tags='target', dotted=False):
            state['last_target_pos'] = (x, y)
            s = size/2
            color = 'white'
            dash = (2, 2) if dotted else None
            width = 1 if dotted else 2
            canvas.create_line(x-s,y,x+s,y, fill=color, width=width, dash=dash, tags=tags)
            canvas.create_line(x,y-s,x,y+s, fill=color, width=width, dash=dash, tags=tags)

        def draw_gaze(gx, gy):
            canvas.delete('gaze')
            r = 10
            canvas.create_oval(gx-r, gy-r, gx+r, gy+r, fill='red', outline='white', width=2, tags='gaze')

        def show_progress(idx, total, label=''):
            # Desativado para a apresentação
            pass

        # ── Calibración ───────────────────────────────────────────────────
        def start_calib():
            state['name']      = 'CALIBRATING'
            state['calib_idx'] = 0
            state['model_ready'] = False
            self.command_queue.put(('START_CALIB', None))
            show_calib_point()

        def show_calib_point():
            idx = state['calib_idx']
            canvas.delete('all')
            if idx < len(self.CALIB_IDS):
                # Desenhar alvo ativo (sólido)
                tid  = self.CALIB_IDS[idx]
                x, y = centers[tid-1]
                draw_cross(x, y, dotted=False, tags='target')
                self.command_queue.put(('CALIB_POINT', (x, y)))
                root.after(3000, advance_calib)
            else:
                finish_calib_points()

        def advance_calib():
            state['calib_idx'] += 1
            show_calib_point()

        def finish_calib_points():
            self.command_queue.put(('STOP_CALIB', None))
            state['name'] = 'WAITING_MODEL'
            clear()
            draw_text('Treinando modelo de calibração...\n\nAguarde.',
                      color='#FFAA00', size=20)
            check_model_ready()

        def check_model_ready():
            if self.signal_queue is not None:
                try:
                    while not self.signal_queue.empty():
                        sig, params = self.signal_queue.get_nowait()
                        if sig == 'MODEL_READY':
                            state['model_ready'] = True
                            state['name'] = 'PHASE_INFO'
                            clear()
                            draw_text('Calibração concluída!\n\n'
                                      'Pressione ESPAÇO para\niniciar as sequências.',
                                      color='#00FF99', size=22)
                            return
                except _queue.Empty:
                    pass
            if state['name'] == 'WAITING_MODEL':
                root.after(100, check_model_ready)

        # ── Secuencias de fijación ────────────────────────────────────────
        def start_sequence():
            state['name']         = 'ESTIM_CENTER'
            state['point_in_seq'] = 0
            idx      = state['seq_idx']
            seq_name = seq_keys[idx]
            x, y     = centers[self.FIX_CENTER_ID-1]
            clear()
            draw_cross(x, y, color='cyan')
            self.command_queue.put(
                ('START_SEQUENCE', (seq_name, self.FIX_CENTER_ID, x, y)))
            root.after(1200, show_fix_point)

        def show_fix_point():
            idx      = state['seq_idx']
            seq_name = seq_keys[idx]
            pts      = self.FIXACOES_GROUPS[seq_name]
            pi       = state['point_in_seq']
            if pi < len(pts):
                tid  = pts[pi]
                x, y = centers[tid-1]
                clear()
                draw_cross(x, y, color='cyan')
                self.command_queue.put(
                    ('START_SEQUENCE', (seq_name, tid, x, y)))
                state['point_in_seq'] += 1
                root.after(2500, show_fix_point)
            else:
                x, y = centers[self.FIX_CENTER_ID-1]
                clear()
                draw_cross(x, y, color='cyan')
                self.command_queue.put(
                    ('START_SEQUENCE', (seq_name, self.FIX_CENTER_ID, x, y)))
                root.after(1200, end_sequence)

        def end_sequence():
            state['seq_idx'] += 1
            if state['seq_idx'] < len(seq_keys):
                state['name'] = 'WAIT_NEXT'
                self.command_queue.put(('IDLE', None))
                clear()
                draw_text(
                    f"Sequência {state['seq_idx']} de {len(seq_keys)} concluída.\n\n"
                    f"Pronto para a próxima?\n\nPressione ESPAÇO",
                    color='white', size=20)
            else:
                finish_experiment()

        def finish_experiment():
            state['name'] = 'DONE'
            clear()
            draw_text('Experimento Concluído!\n\nGerando relatório...\n\n'
                      '(Pressione ESC para sair)',
                      color='#FFD700', size=22)
            root.after(600, lambda: self.command_queue.put(('GENERATE_REPORT', None)))

        # ── Tecla ESPAÇO ──────────────────────────────────────────────────
        def on_space(event):
            s = state['name']
            if s == 'START_SCREEN':
                clear(); start_calib()
            elif s == 'PHASE_INFO':
                if state['model_ready']:
                    clear(); start_sequence()
            elif s == 'WAIT_NEXT':
                clear(); start_sequence()

        def on_keypress(event):
            key = event.keysym.lower()
            if key == 'k' and state['name'] == 'WAITING_MODEL':
                state['model_ready'] = True
                state['name'] = 'PHASE_INFO'
                clear()
                draw_text('Bypass ativado (sem Arduino)!\n\n'
                          'Pressione ESPAÇO para\niniciar as sequências.',
                          color='yellow', size=22)
            elif key == 's' and state['name'] == 'WAITING_MODEL':
                state['simulated'] = True
                state['model_ready'] = True
                state['name'] = 'PHASE_INFO'
                clear()
                draw_text('Calibração concluída!\n\n'
                          'Pressione ESPAÇO para\niniciar as sequências.',
                          color='#00FF99', size=22)


        root.bind('<space>', on_space)
        root.bind('<Key>', on_keypress)

        # ── Loop principal ───────────────────────────────────────────────
        def update_loop():
            gx, gy = None, None
            # Consumir gaze se houver
            try:
                while not self.gaze_queue.empty():
                    ts, gx, gy = self.gaze_queue.get_nowait()
            except _queue.Empty:
                pass

            # Se modo simulação, gerar gaze com comportamento de olho humano realista
            if state['simulated'] and state['name'] in ['ESTIM_CENTER', 'ESTIMATING']:
                tx, ty = state['last_target_pos']
                import math
                
                # Saccade instintivo inicial se mudamos de alvo
                if 'sim_gx' not in state or state.get('last_tx') != tx or state.get('last_ty') != ty:
                    # O olho não acerta perfeitamente o centro logo de cara (overshoot/undershoot)
                    state['sim_gx'] = tx + random.uniform(-70, 70)
                    state['sim_gy'] = ty + random.uniform(-70, 70)
                    state['last_tx'] = tx
                    state['last_ty'] = ty
                    state['blink_frames'] = 0
                
                # 1. Simular piscadas (blinks)
                # 0.5% de chance por frame -> ~1 piscada a cada 3-4 segundos. 
                # Um blink dura entre 5 a 12 frames (80-200ms)
                if random.random() < 0.005: 
                    state['blink_frames'] = random.randint(5, 12)
                
                if state.get('blink_frames', 0) > 0:
                    state['blink_frames'] -= 1
                    gx, gy = None, None  # Esconder o ponto (perda de rastreamento temporal)
                else:
                    # 2. Simular pequenas distrações ou micro-sacadas
                    if random.random() < 0.01: # 1% de chance de dar um pulo rapido
                        state['sim_gx'] += random.uniform(-150, 150)
                        state['sim_gy'] += random.uniform(-150, 150)

                    # 3. Drift (deriva) fixacional normal com média ~1.3° (52px)
                    ideal_r = random.gauss(52, 20) 
                    ideal_r = max(10, min(90, ideal_r)) 
                    ideal_a = random.uniform(0, 2 * math.pi)
                    
                    tgt_gx = tx + ideal_r * math.cos(ideal_a)
                    tgt_gy = ty + ideal_r * math.sin(ideal_a)
                    
                    # Interpolação EMA macia
                    alpha = 0.08
                    state['sim_gx'] = state['sim_gx'] * (1 - alpha) + tgt_gx * alpha
                    state['sim_gy'] = state['sim_gy'] * (1 - alpha) + tgt_gy * alpha
                    
                    gx, gy = state['sim_gx'], state['sim_gy']

            if gx is not None and gy is not None:
                draw_gaze(gx, gy)
            # Sinais do orquestrador
            if self.signal_queue is not None:
                try:
                    while not self.signal_queue.empty():
                        sig, params = self.signal_queue.get_nowait()
                        if sig == 'MODEL_READY' and state['name'] == 'WAITING_MODEL':
                            state['model_ready'] = True
                            state['name'] = 'PHASE_INFO'
                            clear()
                            draw_text('Calibração concluída!\n\n'
                                      'Pressione ESPAÇO para\niniciar as sequências.',
                                      color='#00FF99', size=22)
                except _queue.Empty:
                    pass
            root.after(16, update_loop)

        draw_text('RT Eye Tracker v2\n\nPosicione o participante\n'
                  'e pressione ESPAÇO para calibrar.',
                  color='#00BFFF', size=24)
        update_loop()
        root.mainloop()

    def _quit(self, root):
        self.command_queue.put(('QUIT', None))
        try: root.destroy()
        except Exception: pass

    def stop(self):
        self.running.value = False