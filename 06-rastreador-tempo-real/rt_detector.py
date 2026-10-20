"""
rt_detector_v2.py
=================
Módulo de processamento de imagem e detecção de pupila/glints.

Responsabilidades:
  - Classificação de frames em Brilhante/Escura (sincronização).
  - Segmentação da pupila por watershed e ajuste de círculo.
  - Detecção e seleção de reflexos corneais (glints).
  - Cálculo do vetor de olhar (pupila-glint) com filtragem EMA.
  - Análise de integridade da sincronia (SyncAnalyzer).
"""

import multiprocessing as mp
import time
import math
import queue as _queue
import numpy as np
from collections import defaultdict


def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _midpoint(p1, p2):
    if p1 is None or p2 is None:
        return -1.0, -1.0
    return (p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0

def _iou_rects(r1, r2):
    ix1=max(r1[0],r2[0]); iy1=max(r1[1],r2[1])
    ix2=min(r1[2],r2[2]); iy2=min(r1[3],r2[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    a1=(r1[2]-r1[0])*(r1[3]-r1[1]); a2=(r2[2]-r2[0])*(r2[3]-r2[1])
    return inter/(a1+a2-inter)

def _mask_bbox(mask):
    import cv2
    cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x,y,w,h=cv2.boundingRect(np.vstack(cnts))
    return (x,y,x+w,y+h)


# ═══════════════════════════════════════════════════════════
# ANALIZADOR DE SINCRONÍA — CORRECTO
# ═══════════════════════════════════════════════════════════
class SyncAnalyzer:
    """
    Analiza la secuencia B/D con la definición CORRECTA de glitch.

    Glitch real = B->B o D->D (mismo tipo consecutivo).
    Transición normal = B->D o D->B (NO es glitch).
    """
    RACHA_LARGA_THR = 5   # rachas de este tamaño o mayor se detallan

    def __init__(self):
        self.total      = 0
        self.glitches   = 0
        self.last_type  = None
        self.racha_act  = 0       # longitud de la racha actual del mismo tipo
        self.racha_maxB = 0
        self.racha_maxD = 0
        self.conteoB    = 0
        self.conteoD    = 0
        self.t0         = time.perf_counter()
        self.hist       = defaultdict(int)   # {longitud_racha: conteo}
        self.largas     = []                 # [(tipo, longitud)]
        self._tipo_act  = None

    def update(self, ftype):
        """Registra un frame. Retorna True si es glitch real."""
        self.total += 1
        if ftype == 'brilhante': self.conteoB += 1
        else:                    self.conteoD += 1

        if self.last_type is None:
            self.racha_act = 1
            self._tipo_act = ftype
            self.last_type = ftype
            return False

        if ftype == self.last_type:
            # GLITCH REAL: mismo tipo dos veces seguidas
            self.glitches  += 1
            self.racha_act += 1
            if ftype == 'brilhante':
                self.racha_maxB = max(self.racha_maxB, self.racha_act)
            else:
                self.racha_maxD = max(self.racha_maxD, self.racha_act)
            self.last_type = ftype
            return True
        else:
            # Transición normal: cerrar racha anterior
            if self.racha_act > 1:
                self.hist[self.racha_act] += 1
                if self.racha_act >= self.RACHA_LARGA_THR:
                    self.largas.append((self._tipo_act, self.racha_act))
            self.racha_act = 1
            self._tipo_act = ftype
            self.last_type = ftype
            return False

    @property
    def fps(self):
        e = time.perf_counter() - self.t0
        return self.total / e if e > 0 else 0.0

    @property
    def tasa_pct(self):
        if self.total < 2: return 0.0
        return 100.0 * self.glitches / (self.total - 1)

    @property
    def ratio_BD(self):
        t = self.conteoB + self.conteoD
        if t == 0: return 0.0, 0.0
        return 100.0*self.conteoB/t, 100.0*self.conteoD/t

    def reporte_corto(self):
        pB, pD = self.ratio_BD
        return (f"SYNC fps={self.fps:.1f} "
                f"glitch_real={self.tasa_pct:.1f}% "
                f"maxB={self.racha_maxB} maxD={self.racha_maxD} "
                f"B={pB:.0f}%/D={pD:.0f}%")

    def reporte_completo(self):
        pB, pD = self.ratio_BD
        lines = [
            "\n" + "="*55,
            "  DIAGNÓSTICO DE SINCRONÍA REAL",
            "="*55,
            f"  Frames totales   : {self.total}",
            f"  FPS estimado      : {self.fps:.2f}",
            f"  Distribución B/D  : {pB:.1f}% / {pD:.1f}%  (ideal 50/50)",
            "",
            "  GLITCHES REALES (B→B o D→D):",
            f"    Total glitches  : {self.glitches}",
            f"    Tasa de glitch  : {self.tasa_pct:.2f}%",
            f"    Racha máx B→B  : {self.racha_maxB} frames",
            f"    Racha máx D→D  : {self.racha_maxD} frames",
            "",
        ]

        if self.tasa_pct == 0:
            lines.append("  ✓ SINCRONÍA PERFECTA — cero glitches reales")
        elif self.tasa_pct < 2:
            lines.append("  ✓ SINCRONÍA BUENA — tasa < 2%")
        elif self.tasa_pct < 8:
            lines.append("  ⚠ SINCRONÍA MODERADA — drift ocasional")
            lines.append("    → Prueba ajustar pre_delay_us ±500µs")
        else:
            lines.append("  ✗ SINCRONÍA MALA — muchos glitches reales")
            lines.append("    → Revisar periodo Arduino y pre_delay_us")

        if self.hist:
            lines.append("")
            lines.append("  Histograma de rachas largas:")
            for lng in sorted(self.hist.keys()):
                bar = "█" * min(self.hist[lng], 40)
                lines.append(f"    {lng:3d} frames: {self.hist[lng]:4d}  {bar}")

        if self.largas:
            lines.append("")
            lines.append(f"  Rachas ≥{self.RACHA_LARGA_THR} frames detectadas:")
            for tipo, lng in self.largas[-15:]:
                lines.append(f"    {tipo.upper():12s}  {lng:4d} frames seguidos")

        lines.append("="*55)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# CLASIFICADOR DARK / BRIGHT
# ═══════════════════════════════════════════════════════════
class PupilTypeClassifier:
    """
    Clasificador RELATIVO: no usa umbral absoluto.

    Problema con umbral absoluto (threshold=4):
      - Si los LEDs ON-AXIS son débiles, sus frames tienen pocos píxeles >200DN
      - El umbral los clasifica como ESCURA aunque sean brilhante
      - Resultado: rachas largas D→D artificiales

    Solución RELATIVA:
      Fase bootstrap (primeros N frames sin ROI):
        Usa el histograma completo y guarda el conteo del frame anterior.
        El frame con MÁS píxeles brillantes = brilhante.
        El frame con MENOS píxeles brillantes = escura.
        Esto es robusto: si ambos tienen pocos px brillantes, el que tiene
        MÁS sigue siendo el brilhante.

      Una vez que tenemos ROI de glints:
        Histograma solo en la región del ojo (más preciso y rápido).
        Misma lógica relativa.

    El threshold_abs sigue disponible como fallback opcional
    para cuando no hay frame anterior.
    """

    def __init__(self):
        self._last_count = None   # conteo del frame anterior

    def classify_window(self, img_gray, min_bright_pixel_value=200,
                        glint_roi_mask=None, window=4):
        """
        Clasificador por VENTANA DE FRAMES.

        Problema del clasificador frame-a-frame:
          La cámara a 51fps captura 2 frames por ciclo Arduino (16.7ms).
          Al comparar frame_N con frame_N-1, si ambos son del mismo ciclo
          (ambos brilhante o ambos escura), la comparación falla.

        Solución con ventana:
          Mantiene los últimos N counts en un buffer circular.
          Decisión: si el count actual > mediana del buffer → brilhante.
          Esto es robusto a 1-2 frames duplicados porque la mediana
          no cambia con una o dos anomalías.

          Adicionalmente: si el count actual es el MAYOR de todos los
          de la ventana → claramente brilhante.
          Si es el MENOR → claramente escura.
        """
        import cv2
        if glint_roi_mask is not None and np.any(glint_roi_mask):
            region = cv2.bitwise_and(img_gray, img_gray, mask=glint_roi_mask)
            hist   = cv2.calcHist([region], [0], glint_roi_mask, [256], [0, 256])
        else:
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

        count = int(np.sum(hist[min_bright_pixel_value:256]))

        if not hasattr(self, '_window_buf'):
            self._window_buf = []

        self._window_buf.append(count)
        if len(self._window_buf) > window:
            self._window_buf.pop(0)

        if len(self._window_buf) < 2:
            return 'escura', count

        buf = self._window_buf
        median = sorted(buf)[len(buf)//2]

        # Si el count actual está CLARAMENTE por encima de la mediana → brilhante
        # Si está CLARAMENTE por debajo → escura
        if count > median:
            tipo = 'brilhante'
        else:
            tipo = 'escura'

        return tipo, count

    @staticmethod
    def classify(img_gray, min_bright_pixel_value=200, bright_threshold=4,
                 glint_roi_mask=None):
        """Método estático original — mantenido para compatibilidad."""
        import cv2
        if glint_roi_mask is not None and np.any(glint_roi_mask):
            region = cv2.bitwise_and(img_gray, img_gray, mask=glint_roi_mask)
            hist   = cv2.calcHist([region], [0], glint_roi_mask, [256], [0, 256])
        else:
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        count = int(np.sum(hist[min_bright_pixel_value:256]))
        return ('brilhante' if count >= bright_threshold else 'escura'), count


# ═══════════════════════════════════════════════════════════
# AJUSTE DE CÍRCULO
# ═══════════════════════════════════════════════════════════
class CircleFitter:
    @staticmethod
    def fit_circle(points):
        if points is None or len(points) < 3: return None
        from scipy.optimize import least_squares
        x = points[:,0].astype(float); y = points[:,1].astype(float)
        def res(p,xd,yd): return np.sqrt((xd-p[0])**2+(yd-p[1])**2)-p[2]
        cx0,cy0 = x.mean(),y.mean()
        r0 = np.mean(np.sqrt((x-cx0)**2+(y-cy0)**2))
        try:
            r = least_squares(res,[cx0,cy0,r0],args=(x,y),
                              bounds=([0,0,1],np.inf),xtol=1e-3)
            return float(r.x[0]),float(r.x[1]),float(r.x[2])
        except: return None


# ═══════════════════════════════════════════════════════════
# DETECTOR DE GLINTS
# ═══════════════════════════════════════════════════════════
class GlintDetector:
    @staticmethod
    def detect(img_gray, roi_mask, is_dark, cfg):
        import cv2
        if img_gray is None: return [],[],[]
        proc = img_gray.copy()
        if roi_mask is not None and np.any(roi_mask):
            proc = cv2.bitwise_and(img_gray,img_gray,mask=roi_mask)
        _,mx,_,_ = cv2.minMaxLoc(proc)
        if mx < 50: return [],proc,np.zeros_like(proc)
        gmin=cfg.get('glint_min_area',0); gmax=cfg.get('glint_max_area',500)
        pcts=[0.80,0.72,0.65,0.55,0.45] if is_dark else [0.95,0.90,0.85,0.80,0.70]
        tgt=2 if is_dark else 1
        best=[]; bthr=np.zeros_like(proc)
        k1=np.ones((1,1),np.uint8); k2=np.ones((2,2),np.uint8)
        for pct in pcts:
            tv=max(80,min(240,int(mx*pct)))
            _,thr=cv2.threshold(proc,tv,255,cv2.THRESH_BINARY)
            thr=cv2.morphologyEx(thr,cv2.MORPH_CLOSE,k1,iterations=1)
            thr=cv2.morphologyEx(thr,cv2.MORPH_OPEN, k2,iterations=1)
            cp,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if is_dark and len(cp)==1:
                ap=cv2.contourArea(cp[0])
                if 10<ap<gmax*3: thr=GlintDetector._ws(proc,thr)
            cs,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            gs=[]
            for c in cs:
                a=cv2.contourArea(c)
                if gmin<a<gmax:
                    p=cv2.arcLength(c,True)
                    if p>0:
                        ci=(4*math.pi*a)/(p*p); M=cv2.moments(c)
                        if M["m00"]!=0:
                            gs.append({'center':(M["m10"]/M["m00"],M["m01"]/M["m00"]),
                                       'area':a,'circularity':ci})
            if len(gs)>len(best): best=gs; bthr=thr
            if len(best)>=tgt: break
        return best,proc,bthr

    @staticmethod
    def _ws(gray,tb):
        import cv2
        k=np.ones((3,3),np.uint8)
        sb=cv2.dilate(tb,k,iterations=2)
        dt=cv2.distanceTransform(tb,cv2.DIST_L2,5)
        if dt.max()==0: return tb
        _,sf=cv2.threshold(dt,0.30*dt.max(),255,0)
        sf=np.uint8(sf); un=cv2.subtract(sb,sf)
        _,m=cv2.connectedComponents(sf); m=m+1; m[un==255]=0
        cv2.watershed(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),m)
        out=np.zeros_like(tb); out[m>1]=255
        return out


# ═══════════════════════════════════════════════════════════
# SELECTOR DE GLINTS
# ═══════════════════════════════════════════════════════════
class GlintSelector:
    @staticmethod
    def select_bright(pupil_center, candidates, pupil_mask=None):
        if not pupil_center or not candidates: return []
        if pupil_mask is not None and np.any(pupil_mask):
            f=[g for g in candidates if _om(g['center'],pupil_mask)]
            if f: candidates=f
        candidates=[g for g in candidates if g.get('circularity',0)>=0.70]
        if not candidates: return []
        candidates.sort(key=lambda g:-g.get('circularity',0))
        fin=candidates[:2]
        if len(fin)<=1: return fin
        px,py=pupil_center
        fin.sort(key=lambda g:(g['center'][0]-px)**2+(g['center'][1]-py)**2)
        return fin[:1]

    @staticmethod
    def select_dark(pupil_center, candidates):
        if not pupil_center or not candidates: return []
        candidates.sort(key=lambda g:_dist(g['center'],pupil_center))
        g1=candidates[0]
        if len(candidates)==1: return [g1]
        rest=candidates[1:]; rest.sort(key=lambda g:_dist(g['center'],g1['center']))
        return [g1,rest[0]]

def _om(c,m):
    ix,iy=int(c[0]),int(c[1])
    if 0<=iy<m.shape[0] and 0<=ix<m.shape[1]: return m[iy,ix]==0
    return False


# ═══════════════════════════════════════════════════════════
# SEGMENTADOR
# ═══════════════════════════════════════════════════════════
class PupilSegmenter:
    def __init__(self,cfg):
        self.cfg=cfg
        self.ksm=np.ones((3,3),np.uint8)
        self.kmd=np.ones((5,5),np.uint8)

    def refine_roi(self,eg,bg,fallback_roi=None,fallback_diff=None,
                   prev_bbox=None,frame_count=0):
        import cv2; cfg=self.cfg; H,W=eg.shape[:2]
        diff=cv2.absdiff(bg,eg); ufb=False
        if fallback_diff is not None and cv2.mean(diff)[0]<10:
            diff=fallback_diff.copy(); ufb=True
        if ufb:
            roi=fallback_roi.copy() if (fallback_roi is not None and np.any(fallback_roi)) else None
            return roi,diff,roi,True
        bl=cv2.GaussianBlur(diff,(9,9),0)
        _,rr=cv2.threshold(bl,cfg.get('roi_threshold',80),255,cv2.THRESH_BINARY)
        if rr is None or not np.any(rr):
            return (fallback_roi,diff,fallback_roi,False) if fallback_roi is not None else (None,diff,None,False)
        rr=cv2.morphologyEx(rr,cv2.MORPH_CLOSE,self.kmd,iterations=2)
        rr=cv2.morphologyEx(rr,cv2.MORPH_OPEN, self.kmd,iterations=1)
        n,lb,st,ce=cv2.connectedComponentsWithStats(rr,8,cv2.CV_32S)
        if n<=1:
            return (fallback_roi,diff,fallback_roi,False) if fallback_roi is not None else (None,diff,None,False)
        mn=cfg.get('roi_min_area',300); mx=cfg.get('roi_max_area',40000)
        if frame_count<cfg.get('bootstrap_frames',200) or prev_bbox is None:
            wx1=int(W*0.20);wx2=int(W*0.80);wy1=int(H*0.15);wy2=int(H*0.85)
            bi,bs=-1,-1.0
            for i in range(1,n):
                a=st[i,cv2.CC_STAT_AREA]
                if not(mn<=a<=mx): continue
                bx=int(ce[i][0]);by=int(ce[i][1])
                if not(wx1<=bx<=wx2 and wy1<=by<=wy2): continue
                bm2=np.uint8(lb==i)*255
                if cv2.mean(bg,mask=bm2)[0]>160: continue
                if a>bs: bs=a;bi=i
        else:
            bi,biou=-1,cfg.get('iou_min',0.05)
            for i in range(1,n):
                a=st[i,cv2.CC_STAT_AREA]
                if not(mn<=a<=mx): continue
                x1c=st[i,cv2.CC_STAT_LEFT];y1c=st[i,cv2.CC_STAT_TOP]
                x2c=x1c+st[i,cv2.CC_STAT_WIDTH];y2c=y1c+st[i,cv2.CC_STAT_HEIGHT]
                iou=_iou_rects(prev_bbox,(x1c,y1c,x2c,y2c))
                if iou>biou: biou=iou;bi=i
        if bi==-1:
            return (fallback_roi,diff,fallback_roi,False) if fallback_roi is not None else (None,diff,None,False)
        nr=np.zeros_like(rr); nr[lb==bi]=255
        if cv2.mean(bg,mask=nr)[0]>190:
            return (fallback_roi,diff,fallback_roi,False) if fallback_roi is not None else (None,diff,None,False)
        return nr,diff,nr.copy(),False

    @staticmethod
    def _rg(img,gl):
        import cv2; out=img.copy()
        for g in gl:
            ix,iy=int(g['center'][0]),int(g['center'][1])
            r=int(max(1,math.sqrt(g['area']/math.pi)*2.0))
            cv2.circle(out,(ix,iy),r,0,-1)
        return out

    @staticmethod
    def _ws(binary,bgr):
        import cv2; k=np.ones((3,3),np.uint8)
        op=cv2.morphologyEx(binary,cv2.MORPH_OPEN,k,iterations=2)
        sb=cv2.dilate(op,k,iterations=3)
        dt=cv2.distanceTransform(op,cv2.DIST_L2,5)
        if dt.max()==0: return None,np.zeros_like(binary)
        _,sf=cv2.threshold(dt,0.5*dt.max(),255,0); sf=np.uint8(sf)
        un=cv2.absdiff(sb,sf); _,m=cv2.connectedComponents(sf)
        m=m+1; m[un==255]=0; mf=cv2.watershed(bgr.copy(),m.copy())
        mask=np.zeros_like(binary); mask[mf>1]=255
        return mf,mask

    def _seg_dark(self,g,roi,bgr):
        import cv2; cfg=self.cfg; pm=np.zeros_like(g); pi=[]
        _,inv=cv2.threshold(g,50,255,cv2.THRESH_BINARY_INV)
        ir=cv2.bitwise_and(inv,inv,mask=roi)
        mf,ma=self._ws(ir,bgr)
        if mf is None: return bgr.copy(),pm,[],True
        ri=bgr.copy(); bm=(mf==-1); bm[0,:]=bm[-1,:]=bm[:,0]=bm[:,-1]=False
        ri[bm]=[0,0,255]; pm=ma.copy()
        cs,_=cv2.findContours(ma,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a=cv2.contourArea(c); x,y,w,h=cv2.boundingRect(c)
            ar=float(h)/w if w>0 else 0
            if not(0.65<=ar<=1.5): continue
            if not(cfg.get('dark_area_min',200)<a<11000): continue
            p=cv2.arcLength(c,True)
            if p<=0: continue
            ci=(4*math.pi*a)/(p**2)
            if ci<=0.20: continue
            pts=c.squeeze()
            if pts.ndim==1 or pts.shape[0]<3: continue
            fit=CircleFitter.fit_circle(pts)
            if fit is None or fit[2]<8: continue
            q=0.5*ci+0.5*max(0.0,1.0-abs(ar-1.0))
            pi.append({'center':(fit[0],fit[1]),'radius':fit[2],'fit_circle':fit,
                       'contour':c,'quality':q})
        pi.sort(key=lambda p:p['quality'],reverse=True)
        return ri,pm,pi,(len(pi)==0)

    def _seg_bright(self,g,roi,bgr):
        import cv2; cfg=self.cfg; pm=np.zeros_like(g); pi=[]
        rn=cv2.bitwise_and(g,g,mask=roi)
        try: _,mv,_,_=cv2.minMaxLoc(rn,mask=roi)
        except: mv=0
        thr=max(50,min(200,int(mv*0.5))) if mv>0 else 60
        _,th=cv2.threshold(rn,thr,255,cv2.THRESH_BINARY)
        th=cv2.morphologyEx(th,cv2.MORPH_OPEN, self.ksm,iterations=1)
        th=cv2.morphologyEx(th,cv2.MORPH_CLOSE,self.ksm,iterations=1)
        mf,ma=self._ws(th,bgr)
        if mf is None: return bgr.copy(),pm,[],True
        ri=bgr.copy(); bm=(mf==-1); bm[0,:]=bm[-1,:]=bm[:,0]=bm[:,-1]=False
        ri[bm]=[0,0,255]; pm=ma.copy()
        cs,_=cv2.findContours(ma,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            a=cv2.contourArea(c); x,y,w,h=cv2.boundingRect(c)
            ar=float(h)/w if w>0 else 0
            if not(0.40<=ar<=1.8): continue
            if not(cfg.get('bright_area_min',200)<a<20000): continue
            p=cv2.arcLength(c,True)
            if p<=0: continue
            ci=(4*math.pi*a)/(p**2)
            if ci<=cfg.get('bright_circularity_min',0.20): continue
            pts=c.squeeze()
            if pts.ndim==1 or pts.shape[0]<3: continue
            fit=CircleFitter.fit_circle(pts)
            if fit is None or fit[2]<8: continue
            q=0.5*ci+0.5*max(0.0,1.0-abs(ar-1.0))
            pi.append({'center':(fit[0],fit[1]),'radius':fit[2],'fit_circle':fit,
                       'contour':c,'quality':q})
        pi.sort(key=lambda p:p['quality'],reverse=True)
        return ri,pm,pi,(len(pi)==0)


# ═══════════════════════════════════════════════════════════
# PROCESO PRINCIPAL
# ═══════════════════════════════════════════════════════════
class RT_Detector_v2(mp.Process):

    def __init__(self, frame_queue, vector_queue, config):
        super().__init__(daemon=True)
        self.frame_queue  = frame_queue
        self.vector_queue = vector_queue
        self.config       = config
        self.running      = mp.Value('b', True)

    def run(self):
        import cv2
        print("RT_Detector: Iniciado.")
        cfg=self.config; debug=cfg.get('debug',False)
        seg=PupilSegmenter(cfg)
        mbp=cfg.get('min_bright_pixel_value',200)
        bthr=cfg.get('bright_threshold',4)
        qmin=cfg.get('quality_min',0.35)
        ema_a=cfg.get('ema_alpha',0.6)
        use_relative=cfg.get('relative_classifier', True)  # default: clasificador relativo

        pair={'escura':None,'brilhante':None}
        lgr=lGd=None; pb=None; fc=0; fbc=0; cgroi=None
        evx=evy=None
        classifier=PupilTypeClassifier()   # instancia para clasificador relativo
        sync=SyncAnalyzer()

        # ── Estado para fallback de frames iguales ────────────────────────
        # Cuando llegan 2 frames del mismo tipo (glitch de sincronía),
        # el absdiff ≈ 0 y el ROI no se puede calcular.
        # Estrategia en 3 niveles según gravedad:
        #   Nivel 1 (1-4 glitches):  usar el último ROI bueno sin actualizar
        #   Nivel 2 (5-15 glitches): flujo óptico LK sobre el frame escura
        #   Nivel 3 (>15 glitches):  re-bootstrap (ya existía)
        last_dark_gray  = None   # último frame escura real
        last_bright_gray= None   # último frame brilhante real
        lk_pts_prev     = None   # puntos para Lucas-Kanade
        lk_gray_prev    = None   # frame anterior para LK
        consec_glitch   = 0      # glitches consecutivos actuales
        lk_offset_x     = 0.0   # desplazamiento acumulado por LK
        lk_offset_y     = 0.0
        LK_PARAMS = dict(winSize=(15,15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))

        # ── ROI fijo de bootstrap para el clasificador ────────────────────
        # El histograma del frame completo (307200px) enmascara la señal del
        # glint ON-AXIS (~50px = 0.016%). Usar una ventana central reducida
        # (~36000px) hace la señal 8× más prominente durante el bootstrap.
        # Una vez que el ROI del ojo está disponible (cgroi), se usa ese.
        _H, _W = 480, 640
        bootstrap_roi = np.zeros((_H, _W), dtype=np.uint8)
        _bx1 = int(_W * 0.28); _bx2 = int(_W * 0.72)   # 28-72% horizontal
        _by1 = int(_H * 0.22); _by2 = int(_H * 0.68)   # 22-68% vertical
        bootstrap_roi[_by1:_by2, _bx1:_bx2] = 255
        stats={'fr':0,'ve':0,'bl':0,'np':0,'ng':0,'lq':0}
        tl=tsync=time.perf_counter()
        PW,PH=640,480

        if debug:
            WIN='RT_Detector DEBUG — Q=cerrar'
            cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN,1920,988)

        def bl(h=480,w=640): return np.zeros((h,w,3),np.uint8)
        def tb(img,h=480,w=640):
            if img is None: return bl(h,w)
            o=img.copy() if img.ndim==3 else cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            if o.shape[:2]!=(h,w): o=cv2.resize(o,(w,h))
            return o

        try:
            while self.running.value:
                try: ts,fi,frame=self.frame_queue.get(timeout=0.05)
                except _queue.Empty: continue

                stats['fr']+=1; fc+=1
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                # ── Clasificar frame (RELATIVO o absoluto) ───────────────
                # ROI para clasificador: ojo real si disponible, bootstrap si no
                _roi_cls = cgroi if (cgroi is not None and np.any(cgroi)) \
                           else bootstrap_roi
                if use_relative:
                    ftype,bc=classifier.classify_window(gray,mbp,
                                                        glint_roi_mask=_roi_cls,
                                                        window=4)
                else:
                    ftype,bc=PupilTypeClassifier.classify(gray,mbp,bthr,
                                                          glint_roi_mask=_roi_cls)
                # ── CONTADOR CORRECTO DE GLITCHES ─────────────────────────
                is_glitch = sync.update(ftype)
                # (glitch real = B→B o D→D, transición normal B→D no cuenta)

                if is_glitch:
                    consec_glitch += 1
                else:
                    consec_glitch = 0

                # ── FALLBACK MULTINIVEL para frames del mismo tipo ────────
                if is_glitch and consec_glitch <= 4:
                    # Nivel 1: usar el último ROI bueno sin cambios
                    # El par que tenemos es inválido (mismo tipo) pero
                    # el vector EMA del frame anterior sigue siendo válido.
                    # Propagamos el último vector sin enviar uno nuevo.
                    # Esto mantiene el punto rojo visible en la UI.
                    if evx is not None and not self.vector_queue.full():
                        self.vector_queue.put_nowait(
                            (ts, evx, evy,
                             float(pb[0]+pb[2])//2 if pb else 320.0,
                             float(pb[1]+pb[3])//2 if pb else 240.0,
                             0.0, 0.0))
                    pair[ftype]={'gray':gray,'bgr':frame,'ts':ts}
                    if pair['escura'] is None or pair['brilhante'] is None: continue

                elif is_glitch and 5 <= consec_glitch <= 20:
                    # Nivel 2: flujo óptico Lucas-Kanade
                    # Estimamos cuánto se movió la pupila entre el frame actual
                    # y el último frame bueno del mismo tipo.
                    # Desplazamos el último vector EMA con ese offset.
                    if (lk_gray_prev is not None and last_dark_gray is not None
                            and evx is not None and pb is not None):
                        try:
                            # Detectar puntos en el último ROI bueno
                            if lk_pts_prev is None:
                                roi_u8 = lgr if lgr is not None else bootstrap_roi
                                corners = cv2.goodFeaturesToTrack(
                                    last_dark_gray, maxCorners=20, qualityLevel=0.3,
                                    minDistance=7, mask=roi_u8)
                                lk_pts_prev = corners
                            if lk_pts_prev is not None and len(lk_pts_prev) > 0:
                                pts_new, st_lk, _ = cv2.calcOpticalFlowPyrLK(
                                    lk_gray_prev, gray, lk_pts_prev, None, **LK_PARAMS)
                                good_new = pts_new[st_lk==1]
                                good_old = lk_pts_prev[st_lk==1]
                                if len(good_new) > 0:
                                    dx = float(np.median(good_new[:,0] - good_old[:,0]))
                                    dy = float(np.median(good_new[:,1] - good_old[:,1]))
                                    lk_offset_x += dx; lk_offset_y += dy
                                    lk_pts_prev = good_new.reshape(-1,1,2)
                                    lk_gray_prev = gray
                                    # Enviar vector ajustado con el offset LK
                                    vx_lk = evx - lk_offset_x
                                    vy_lk = evy - lk_offset_y
                                    if not self.vector_queue.full():
                                        self.vector_queue.put_nowait(
                                            (ts, vx_lk, vy_lk, 0.0, 0.0, 0.0, 0.0))
                        except Exception:
                            pass
                    pair[ftype]={'gray':gray,'bgr':frame,'ts':ts}
                    if pair['escura'] is None or pair['brilhante'] is None: continue

                else:
                    # Normal o Nivel 3 (>20): resetear LK, dejar que el sistema
                    # intente con lo que tiene
                    if not is_glitch:
                        lk_pts_prev  = None
                        lk_offset_x  = 0.0
                        lk_offset_y  = 0.0
                        lk_gray_prev = gray
                    pair[ftype]={'gray':gray,'bgr':frame,'ts':ts}
                    if pair['escura'] is None or pair['brilhante'] is None: continue

                dg=pair['escura']['gray']; bg=pair['brilhante']['gray']
                db=pair['escura']['bgr'];  bb=pair['brilhante']['bgr']
                tsp=pair['escura']['ts']

                roi,diff,_,ufb=seg.refine_roi(dg,bg,lgr,lGd,pb,fc)
                if roi is None or not np.any(roi):
                    fbc+=1
                    if fbc>cfg.get('fallback_limit',30):
                        pb=None;fbc=0;cgroi=None
                    # ── Flujo óptico como fallback cuando ROI falla ───────
                    # Si tenemos el último ROI bueno y el frame anterior,
                    # usar Lucas-Kanade para estimar cuánto se movió la pupila
                    # y propagar el último vector EMA válido (sin actualizar el ROI)
                    # Esto mantiene el punto rojo visible durante drops cortos
                    # en vez de desaparecer completamente
                    if lgr is not None and np.any(lgr) and evx is not None:
                        # No generamos nuevo vector (podría ser incorrecto)
                        # pero tampoco reseteamos el EMA — el último vector
                        # bueno sigue siendo mejor que nada
                        pass   # el vector EMA ya se enviará si status=1 más abajo
                    pair={'escura':None,'brilhante':None}; continue
                fbc=0
                kg=np.ones((12,12),np.uint8)
                gr=cv2.dilate(roi,kg,iterations=4); cgroi=gr

                gld,bwd,thd=GlintDetector.detect(dg,gr,True,cfg)
                glb,bwb,thb=GlintDetector.detect(bg,gr,False,cfg)
                dc=PupilSegmenter._rg(dg,gld); bc2=PupilSegmenter._rg(bg,glb)
                rd,pmd,psd,bld=seg._seg_dark(dc,roi,db)
                rb,pmb,psb,blb=seg._seg_bright(bc2,roi,bb)

                if not bld and psd:
                    lgr=roi; lGd=diff; pb=_mask_bbox(roi)
                    last_dark_gray   = dg.copy()
                    last_bright_gray = bg.copy()
                    lk_pts_prev      = None   # resetear LK al tener ROI nuevo
                    lk_offset_x      = 0.0
                    lk_offset_y      = 0.0
                    lk_gray_prev     = dg.copy()

                px=py=gx=gy=vx=vy=0.0; st=0; q=0.0
                pv=None; gv=[]; gs='-'

                if psd and psd[0].get('fit_circle'):
                    cx,cy,r=psd[0]['fit_circle']; px,py=cx,cy
                    q=psd[0].get('quality',0.0); pv=(int(cx),int(cy))
                    gss=GlintSelector.select_dark((px,py),gld)
                    if len(gss)>=2:
                        gx,gy=_midpoint(gss[0]['center'],gss[1]['center'])
                        gv=gss;gs='dark×2';q*=1.0
                    elif len(gss)==1:
                        gx,gy=gss[0]['center'];gv=gss;gs='dark×1';q*=0.75
                    else:
                        gb=GlintSelector.select_bright((px,py),glb,pmd)
                        if gb: gx,gy=gb[0]['center'];gv=gb;gs='bright_fb';q*=0.5
                    if gx!=0.0 or gy!=0.0:
                        vrx=px-gx;vry=py-gy
                        if evx is None: evx,evy=vrx,vry
                        else: evx=ema_a*vrx+(1-ema_a)*evx;evy=ema_a*vry+(1-ema_a)*evy
                        vx,vy=evx,evy
                        if q>=qmin: st=1;stats['ve']+=1
                        else: stats['lq']+=1
                    else: evx=evy=None;stats['ng']+=1
                elif psb and psb[0].get('fit_circle'):
                    cx,cy,r=psb[0]['fit_circle'];px,py=cx,cy
                    q=psb[0].get('quality',0.0)*0.6;pv=(int(cx),int(cy))
                    gb=GlintSelector.select_bright((px,py),glb,pmb)
                    if gb:
                        gx,gy=gb[0]['center'];gv=gb;gs='bright_only'
                        vrx=px-gx;vry=py-gy
                        if evx is None: evx,evy=vrx,vry
                        else: evx=ema_a*vrx+(1-ema_a)*evx;evy=ema_a*vry+(1-ema_a)*evy
                        vx,vy=evx,evy
                        if q>=qmin: st=1;stats['ve']+=1
                        else: stats['lq']+=1
                    else: stats['ng']+=1
                else:
                    evx=evy=None
                    if bld and blb: stats['bl']+=1
                    else: stats['np']+=1

                if st==1 and not self.vector_queue.full():
                    self.vector_queue.put_nowait((tsp,vx,vy,px,py,gx,gy))

                # ── MOSAICO DEBUG ─────────────────────────────────────────
                if debug:
                    f=cv2.FONT_HERSHEY_SIMPLEX
                    p1=tb(db)
                    if pv:
                        cv2.line(p1,(0,pv[1]),(PW,pv[1]),(255,0,0),1)
                        cv2.line(p1,(pv[0],0),(pv[0],PH),(255,0,0),1)
                        cv2.circle(p1,pv,5,(255,0,0),-1)
                    for i,g in enumerate(gv):
                        gc=(int(g['center'][0]),int(g['center'][1]))
                        col=(0,255,255) if i==0 else (0,0,255)
                        cv2.circle(p1,gc,4,col,-1)
                        cv2.line(p1,(0,gc[1]),(PW,gc[1]),col,1)
                        cv2.line(p1,(gc[0],0),(gc[0],PH),col,1)
                    if len(gv)>=2: cv2.circle(p1,(int(gx),int(gy)),7,(0,255,0),2)
                    oc=(0,255,0) if st==1 else (0,0,255)
                    cv2.putText(p1,f"ESCURO v=({vx:.1f},{vy:.1f}) [{gs}] q={q:.2f}",
                                (6,20),f,0.45,oc,1)
                    cv2.putText(p1,f"thr={bthr} px={bc} glitch={sync.tasa_pct:.1f}%",
                                (6,36),f,0.4,(255,255,0),1)
                    p2=tb(bb)
                    if psb and psb[0].get('fit_circle'):
                        bx2,by2,br2=psb[0]['fit_circle']
                        cv2.circle(p2,(int(bx2),int(by2)),int(br2),(0,165,255),2)
                    cv2.putText(p2,f"BRILHANTE pups={len(psb)} gl={len(glb)}",
                                (6,20),f,0.45,(0,200,255),1)
                    p3=tb(diff)
                    if roi is not None and np.any(roi):
                        cr,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(p3,cr,-1,(0,255,0),2)
                    cv2.putText(p3,f"AbsDiff thr={cfg.get('roi_threshold',80)}",
                                (6,PH-10),f,0.4,(0,255,0),1)
                    p4=tb(roi)
                    if roi is not None and np.any(roi):
                        cr2,_=cv2.findContours(roi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(p4,cr2,-1,(0,255,0),2)
                        cv2.putText(p4,f"area={int(np.sum(roi)/255)}px",
                                    (6,20),f,0.4,(255,255,0),1)
                    cv2.putText(p4,"ROI",(6,PH-10),f,0.4,(0,255,0),1)
                    p5=tb(thd) if thd is not None else bl()
                    for g in gld:
                        gc=(int(g['center'][0]),int(g['center'][1]))
                        cv2.circle(p5,gc,6,(0,255,255),2)
                    cv2.putText(p5,f"GlintThr_dark n={len(gld)}",
                                (6,PH-10),f,0.4,(0,255,0),1)
                    p6=tb(dg)
                    if psd and psd[0].get('fit_circle'):
                        ccx,ccy,rr2=psd[0]['fit_circle']
                        cv2.circle(p6,(int(ccx),int(ccy)),int(rr2),(0,255,0),2)
                        cv2.circle(p6,(int(ccx),int(ccy)),4,(0,0,255),-1)
                        if psd[0].get('contour') is not None:
                            cv2.drawContours(p6,[psd[0]['contour']],-1,(255,128,0),1)
                    qw=int(min(q,1.0)*PW); qc=(0,255,0) if q>=qmin else (0,100,255)
                    cv2.rectangle(p6,(0,PH-8),(qw,PH),qc,-1)
                    cv2.putText(p6,f"FitCircle q={q:.2f}",
                                (6,PH-10),f,0.4,(0,255,0),1)
                    for img,lbl in [(p1,"P1:ESCURO"),(p2,"P2:BRILHANTE"),
                                    (p3,"P3:AbsDiff"),(p4,"P4:ROI"),
                                    (p5,"P5:GlintThr"),(p6,"P6:FitCircle")]:
                        cv2.putText(img,lbl,(6,PH-26),f,0.38,(80,80,80),1)
                    top=np.hstack([p1,p2,p3]); bot=np.hstack([p4,p5,p6])
                    mos=np.vstack([top,bot])
                    pB2,pD2=sync.ratio_BD
                    bar=np.zeros((28,mos.shape[1],3),np.uint8)
                    cv2.putText(bar,
                        f"  fr={stats['fr']} ve={stats['ve']} bl={stats['bl']} "
                        f"ng={stats['ng']} | GLITCH_REAL={sync.glitches}"
                        f"({sync.tasa_pct:.1f}%) maxB={sync.racha_maxB}"
                        f" maxD={sync.racha_maxD} fps={sync.fps:.1f}"
                        f" B={pB2:.0f}%/D={pD2:.0f}%",
                        (6,18),f,0.38,(200,200,200),1)
                    mos=np.vstack([mos,bar])
                    cv2.imshow(WIN,mos)
                    k=cv2.waitKey(1)&0xFF
                    if k in(ord('q'),ord('Q')): cv2.destroyWindow(WIN);debug=False

                pair={'escura':None,'brilhante':None}

                # Reportes de diagnóstico desativados para a apresentação
                pass

        finally:
            print(sync.reporte_completo())

        try:
            import cv2 as _c; _c.destroyAllWindows()
        except Exception: pass
        print("RT_Detector: Detenido.")

    def stop(self):
        self.running.value = False