"""
rt_hardware_v2.py
=================
Módulo de captura de vídeo e sincronização com Arduino.

Responsabilidades:
  - Abrir a câmera PS3 Eye e configurar exposição/FPS.
  - Conectar ao Arduino para controle do estroboscópio LED.
  - Sincronização ativa via método "faixa preta" (fases 0-3).
  - Monitoramento dinâmico de glitches de sincronização.
  - Envio dos frames capturados para o detector e preview.
"""

import multiprocessing as mp
import time
import numpy as np


class FaixaPretaDetector:
    KERNEL   = 5
    DARK_THR = 25
    MIN_W    = 2
    MAX_W    = 100

    @staticmethod
    def detect(gray_frame):
        H, W = gray_frame.shape
        row_mean = gray_frame.mean(axis=1).astype(float)
        k        = FaixaPretaDetector.KERNEL
        smoothed = np.convolve(row_mean, np.ones(k)/k, mode='same')
        gm       = smoothed.mean()
        dark_rows= np.where(smoothed < gm - FaixaPretaDetector.DARK_THR)[0]
        if len(dark_rows) < FaixaPretaDetector.MIN_W:
            return None
        best=(0,-1,-1); cs,cl=dark_rows[0],1
        for i in range(1,len(dark_rows)):
            if dark_rows[i]==dark_rows[i-1]+1: cl+=1
            else:
                if cl>best[0]: best=(cl,cs,dark_rows[i-1])
                cs,cl=dark_rows[i],1
        if cl>best[0]: best=(cl,cs,dark_rows[-1])
        ln,bs,be=best
        if ln<FaixaPretaDetector.MIN_W or ln>FaixaPretaDetector.MAX_W: return None
        return float(bs+be)/2.0


class RT_Hardware_v2(mp.Process):

    PS3_VENDOR = '1415'
    PS3_MODEL  = '2000'

    STABILIZE_SECS = 3.0
    FPS_MEAS_SECS  = 2.0
    FINE_STEP      = 1
    FINE_HISTORY   = 8
    CONVERGE_PX    = 1.5
    BIAS_US        = 600
    GLITCH_THR     = 0.25
    GLITCH_WIN     = 240

    def __init__(self, frame_queue, preview_queue, sync_queue, config):
        super().__init__(daemon=True)
        self.frame_queue   = frame_queue
        self.preview_queue = preview_queue
        self.sync_queue    = sync_queue
        self.config        = config
        self.running       = mp.Value('b', True)

    def _find_ps3_eye(self):
        try:
            import pyudev
            ctx = pyudev.Context()
            for dev in ctx.list_devices(subsystem='video4linux'):
                p = dev.properties
                if (p.get('ID_VENDOR_ID') == self.PS3_VENDOR and
                        p.get('ID_MODEL_ID') == self.PS3_MODEL):
                    node = dev.device_node
                    idx  = int(node.replace('/dev/video', ''))
                    print(f"RT_Hardware: PS3 Eye encontrada en {node} (index={idx})")
                    return idx
        except Exception as e:
            print(f"RT_Hardware: pyudev no disponible ({e}).")
        return 0

    def _open_camera(self, cam_id):
        import cv2
        idx = self._find_ps3_eye() if cam_id == 'auto' else int(cam_id)
        cap = None
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened(): break
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE,     -6)
        w   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"RT_Hardware: Câmera aberta — {w:.0f}×{h:.0f} @ {fps:.0f} fps")
        return cap

    def _open_arduino(self, port):
        if self.config.get('no_arduino'):
            print("RT_Hardware: Modo sin Arduino.")
            return None
        
        # Ajuste para Windows: se o porto for o padrão Linux e estivermos no Windows,
        # provavelmente não vai funcionar.
        import os
        if os.name == 'nt' and port.startswith('/dev/'):
            print(f"RT_Hardware: Aviso — Porto Linux '{port}' ignorado no Windows.")
            return None

        try:
            import serial
            ser = serial.Serial(port, 115200, timeout=1)
            time.sleep(2.0)
            print(f"RT_Hardware: Arduino conectado em {port}.")
            return ser
        except Exception as e:
            print(f"RT_Hardware: Arduino não encontrado em {port} ({e})")
            return None


    def _send(self, ser, cmd):
        if ser and ser.is_open:
            ser.write(f'{cmd}\n'.encode())
            ser.flush()

    def _phase0_measure_fps(self, cap):
        import cv2
        print(f"RT_Hardware: [FASE 0] Estabilizando câmera ({self.STABILIZE_SECS}s)...")
        t_end = time.perf_counter() + self.STABILIZE_SECS
        while time.perf_counter() < t_end:
            cap.read()
        print(f"RT_Hardware: [FASE 0] Midiendo FPS real ({self.FPS_MEAS_SECS}s)...")
        count = 0; t0 = time.perf_counter()
        t_end = t0 + self.FPS_MEAS_SECS
        while time.perf_counter() < t_end:
            ret, _ = cap.read()
            if ret: count += 1
        elapsed   = time.perf_counter() - t0
        fps_real  = count / elapsed
        period_us = int(1_000_000 / fps_real)
        print(f"RT_Hardware: FPS real = {fps_real:.2f}  →  periodo = {period_us} µs")
        return fps_real, period_us

    def _phase12_fine_tune(self, cap, ser, period_us):
        import cv2
        print("RT_Hardware: [FASE 1-2] Buscando faixa preta...")
        det=FaixaPretaDetector(); history=[]; period=period_us
        t0=time.perf_counter(); no_band=0
        while time.perf_counter()-t0 < 30.0:
            ret,frame=cap.read()
            if not ret: continue
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            pos=det.detect(gray)
            if pos is None:
                no_band+=1
                if no_band>30:
                    print("RT_Hardware: Faixa no detectada — omitiendo ajuste fino.")
                    return period
                continue
            no_band=0; history.append(pos)
            if len(history)>self.FINE_HISTORY: history.pop(0)
            if len(history)>=self.FINE_HISTORY:
                movement=history[-1]-np.mean(history[:-1])
                if abs(movement)<self.CONVERGE_PX:
                    print(f"RT_Hardware: Periodo estable = {period} µs  (faixa @ {pos:.1f}px)")
                    return period
                period+=self.FINE_STEP if movement>0 else -self.FINE_STEP
                period=max(14000,min(22000,period))
                self._send(ser,f'M{period}')
        print(f"RT_Hardware: Ajuste fino timeout — periodo = {period} µs")
        return period

    def _phase3_move_band(self, cap, ser, stable_period):
        import cv2
        print(f"RT_Hardware: [FASE 3] Moviendo faixa (bias={self.BIAS_US}µs)...")
        det=FaixaPretaDetector(); TARGET_Y=480*0.80
        period_biased=stable_period+self.BIAS_US
        self._send(ser,f'M{period_biased}'); time.sleep(0.5)
        no_band=0
        for _ in range(150):
            ret,frame=cap.read()
            if not ret: continue
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            pos=det.detect(gray)
            if pos is None:
                no_band+=1
                if no_band>=8:
                    self._send(ser,f'M{stable_period}')
                    print(f"RT_Hardware: Faixa desapareció → sincronía activa. periodo={stable_period}µs")
                    return stable_period
            else:
                no_band=0
                if pos>TARGET_Y:
                    self._send(ser,f'M{stable_period}')
                    print(f"RT_Hardware: Faixa en zona segura ({pos:.0f}px). periodo={stable_period}µs")
                    return stable_period
        self._send(ser,f'M{stable_period}')
        print(f"RT_Hardware: Fase 3 timeout — periodo={stable_period}µs")
        return stable_period

    def run(self):
        if self.config.get('no_camera'):
            self._run_simulated(); return

        import cv2
        cap = self._open_camera(self.config.get('camera_id', 'auto'))
        ser = self._open_arduino(self.config.get('arduino_port', '/dev/ttyACM0'))
        if not cap.isOpened():
            print("RT_Hardware: FATAL — câmera não abre."); return

        strobe_dur = self.config.get('strobe_duration_us', 7000)
        pre_delay  = self.config.get('pre_delay_us', 3500)

        if ser:
            self._send(ser, f'S{strobe_dur}')
            self._send(ser, f'P{pre_delay}')
            fps_real, period_us = self._phase0_measure_fps(cap)
            self._send(ser, f'M{period_us}')
            time.sleep(0.3)
            period_us = self._phase12_fine_tune(cap, ser, period_us)
            period_us = self._phase3_move_band(cap, ser, period_us)
            print(f"RT_Hardware: ✓ Sincronía activa. periodo={period_us}µs  (~{1_000_000/period_us:.1f}fps)")
        else:
            period_us = self.config.get('initial_period_us', 16666)

        # Esvaziar frame_queue antes de iniciar captura contínua
        drained = 0
        while not self.frame_queue.empty():
            try: self.frame_queue.get_nowait(); drained+=1
            except Exception: break
        if drained > 0:
            print(f"RT_Hardware: Frame queue esvaziado ({drained} frames obsoletos)")

        print("RT_Hardware: Bucle de captura iniciado.")
        frame_index  = 0
        last_type_hw = None
        glitch_win   = []
        reajust_cnt  = 0
        mbp          = self.config.get('min_bright_pixel_value', 200)
        bthr         = self.config.get('bright_threshold', 4)

        while self.running.value:
            ret, frame = cap.read()
            if not ret: time.sleep(0.005); continue

            ts = time.perf_counter(); frame_index += 1

            if not self.frame_queue.full():
                self.frame_queue.put_nowait((ts, frame_index, frame))
            if not self.preview_queue.full():
                self.preview_queue.put_nowait((ts, frame_index, frame.copy()))

            # Monitoramento de glitches na sincronização
            if ser:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist  = cv2.calcHist([gray], [0], None, [256], [0, 256])
                count = int(np.sum(hist[mbp:256]))
                is_br = count >= bthr
                if last_type_hw is not None:
                    glitch_win.append(is_br == last_type_hw)
                    if len(glitch_win) > self.GLITCH_WIN: glitch_win.pop(0)
                    if len(glitch_win) >= self.GLITCH_WIN:
                        rate = sum(glitch_win)/len(glitch_win)
                        if rate > self.GLITCH_THR:
                            reajust_cnt += 1
                            delta       = 1 if reajust_cnt%2==0 else -1
                            period_us   = max(14000, min(22000, period_us+delta))
                            self._send(ser, f'M{period_us}')
                            glitch_win.clear()
                last_type_hw = is_br

            while not self.sync_queue.empty():
                try:
                    cmd = self.sync_queue.get_nowait()
                    self._send(ser, cmd)
                except Exception: pass

        cap.release()
        if ser and ser.is_open: ser.close()
        print("RT_Hardware: Detenido.")

    def _run_simulated(self):
        import cv2
        print("RT_Hardware: Modo SIMULADO.")
        fi = 0
        while self.running.value:
            frame = np.zeros((480,640,3),dtype=np.uint8)
            ts=time.perf_counter(); fi+=1
            cy,cx=240,320
            if fi%2==1:
                cv2.circle(frame,(cx,cy),40,(200,200,200),-1)
                cv2.circle(frame,(cx,cy),6,(255,255,255),-1)
            else:
                cv2.circle(frame,(cx,cy),40,(20,20,20),-1)
                cv2.circle(frame,(cx-20,cy),4,(255,255,255),-1)
                cv2.circle(frame,(cx+20,cy),4,(255,255,255),-1)
            if not self.frame_queue.full():
                self.frame_queue.put_nowait((ts,fi,frame))
            if not self.preview_queue.full():
                self.preview_queue.put_nowait((ts,fi,frame.copy()))
            time.sleep(1/60)

    def stop(self):
        self.running.value = False