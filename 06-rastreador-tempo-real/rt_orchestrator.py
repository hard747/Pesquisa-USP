"""
rt_orchestrator_v2.py
=====================
Orquestrador central do RT Eye Tracker v2.

Responsabilidades:
  - Lançar e monitorar os 4 processos (Hardware, Detector, Engine, UI).
  - Rotear comandos da UI para o Engine e Hardware.
  - Coordenar a transição calibração → estimação.

Parâmetros disponíveis:
  --bright-threshold  INT     (def: 4)
  --roi-threshold     INT     (def: 80)
  --quality-min       FLOAT   (def: 0.35)
  --ema-alpha         FLOAT   (def: 0.30)
  --bootstrap-frames  INT     (def: 150)
  --iou-min           FLOAT   (def: 0.05)
  --calib-vectors     INT     (def: 30)
  --max-vector-px     INT     (def: 35)
  --debug
  --no-arduino / --no-camera
"""

import multiprocessing as mp
import time
import sys
import argparse
import signal


DEFAULT_CONFIG = {
    'arduino_port':           '/dev/ttyACM0',
    'camera_id':              'auto',
    'initial_period_us':      16666,
    'strobe_duration_us':     7000,
    'pre_delay_us':           3500,
    'screen_width':           1920,
    'screen_height':          1080,
    # Classificação
    'bright_threshold':       4,
    'min_bright_pixel_value': 200,
    # ROI
    'roi_threshold':          80,
    'roi_min_area':           300,
    'roi_max_area':           40000,
    'bootstrap_frames':       150,
    'iou_min':                0.05,
    'fallback_limit':         30,
    # Glints / pupila
    'glint_min_area':         0,
    'glint_max_area':         500,
    'dark_area_min':          200,
    'bright_area_min':        200,
    'bright_circularity_min': 0.20,
    # Calidad / suavizado / outliers
    'quality_min':            0.35,
    'ema_alpha':              0.30,
    'max_vector_px':          35,
    # Calibración / estimación
    'calib_vectors_per_point': 30,
    'poly_degree':             2,
    'ridge_alpha':             5.0,
    # Física
    'monitor_width_mm':       507.0,
    'monitor_width_px':       1920,
    'distance_mm':            600.0,
    'debug':                  False,
    'relative_classifier':    True,   # clasificador relativo
}


def parse_args():
    p = argparse.ArgumentParser(description='RT Eye Tracker v2')
    p.add_argument('--arduino-port',     default=DEFAULT_CONFIG['arduino_port'])
    p.add_argument('--camera-id',        default=DEFAULT_CONFIG['camera_id'])
    p.add_argument('--bright-threshold', type=int,   default=DEFAULT_CONFIG['bright_threshold'])
    p.add_argument('--min-bright-pixel', type=int,   default=DEFAULT_CONFIG['min_bright_pixel_value'])
    p.add_argument('--roi-threshold',    type=int,   default=DEFAULT_CONFIG['roi_threshold'])
    p.add_argument('--quality-min',      type=float, default=DEFAULT_CONFIG['quality_min'])
    p.add_argument('--ema-alpha',        type=float, default=DEFAULT_CONFIG['ema_alpha'])
    p.add_argument('--bootstrap-frames', type=int,   default=DEFAULT_CONFIG['bootstrap_frames'])
    p.add_argument('--iou-min',          type=float, default=DEFAULT_CONFIG['iou_min'])
    p.add_argument('--calib-vectors',    type=int,   default=DEFAULT_CONFIG['calib_vectors_per_point'])
    p.add_argument('--max-vector-px',    type=int,   default=DEFAULT_CONFIG['max_vector_px'])
    p.add_argument('--pre-delay',        type=int,   default=DEFAULT_CONFIG['pre_delay_us'],
                   help='Pre-delay en µs antes del pulso LED (def: 3500, prueba 0)')
    p.add_argument('--strobe-duration',  type=int,   default=DEFAULT_CONFIG['strobe_duration_us'],
                   help='Duración del pulso LED en µs (def: 7000)')
    p.add_argument('--debug',            action='store_true')
    p.add_argument('--no-relative',       action='store_true',
                   help='Usar clasificador absoluto (threshold fijo) en vez del relativo')
    p.add_argument('--no-arduino',       action='store_true')
    p.add_argument('--no-camera',        action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    config = dict(DEFAULT_CONFIG)
    config.update({
        'arduino_port':           args.arduino_port,
        'camera_id':              args.camera_id,
        'bright_threshold':       args.bright_threshold,
        'min_bright_pixel_value': args.min_bright_pixel,
        'roi_threshold':          args.roi_threshold,
        'quality_min':            args.quality_min,
        'ema_alpha':              args.ema_alpha,
        'bootstrap_frames':       args.bootstrap_frames,
        'iou_min':                args.iou_min,
        'calib_vectors_per_point': args.calib_vectors,
        'max_vector_px':          args.max_vector_px,
        'debug':                  args.debug,
        'relative_classifier':    not args.no_relative,
        'pre_delay_us':           args.pre_delay,
        'strobe_duration_us':     args.strobe_duration,
        'no_arduino':             args.no_arduino,
        'no_camera':              args.no_camera,
    })

    print("=" * 62)
    print("  RT Eye Tracker v2  —  Sistema de Rastreamento Ocular")
    print("=" * 62)
    for k, v in [('Arduino', config['arduino_port']),
                 ('Cámara',  config['camera_id']),
                 ('bright_thr (abs)', config['bright_threshold']),
                 ('roi_thr',    config['roi_threshold']),
                 ('quality_min', config['quality_min']),
                 ('ema_alpha',   config['ema_alpha']),
                 ('iou_min',     config['iou_min']),
                 ('max_vec_px',  config['max_vector_px']),
                 ('vecs/punto',  config['calib_vectors_per_point']),
                 ('Debug',       'SÍ' if config['debug'] else 'NO'),
                 ('Clasificador', 'RELATIVO' if config['relative_classifier'] else 'absoluto(thr='+str(config['bright_threshold'])+')')]: 
        print(f"  {k:12s}: {v}")
    print("=" * 62)

    from rt_hardware_v2  import RT_Hardware_v2
    from rt_detector_v2  import RT_Detector_v2
    from rt_engine_v2    import RT_Engine_v2
    from rt_ui_v2        import RT_UI_v2

    # Bandeira compartilhada entre orquestrador e engine
    model_ready_flag = mp.Value('b', False)

    frame_queue   = mp.Queue(maxsize=6)
    preview_queue = mp.Queue(maxsize=4)
    vector_queue  = mp.Queue(maxsize=300)
    gaze_queue    = mp.Queue(maxsize=10)
    engine_cmd    = mp.Queue()
    ui_cmd        = mp.Queue()
    hw_sync       = mp.Queue()
    # Fila para enviar sinais de volta à UI
    ui_signal_queue = mp.Queue()

    p_hw  = RT_Hardware_v2(frame_queue, preview_queue, hw_sync, config)
    p_det = RT_Detector_v2(frame_queue, vector_queue, config)
    p_eng = RT_Engine_v2(vector_queue, gaze_queue, engine_cmd, config,
                         model_ready_flag=model_ready_flag)
    p_ui  = RT_UI_v2(ui_cmd, gaze_queue, preview_queue, config,
                     signal_queue=ui_signal_queue)

    processes = [p_hw, p_det, p_eng, p_ui]

    def shutdown(sig=None, frame=None):
        print("\nOrchestrator: Apagando sistema...")
        for p in processes:
            try: p.stop()
            except Exception: pass
        time.sleep(0.5)
        for p in processes:
            p.terminate(); p.join(timeout=3)
        print("Orchestrator: Limpieza completa.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Orchestrator: Iniciando procesos...")
    for p in processes: p.start()
    print("Orchestrator: Sistema activo. Esperando eventos...\n")

    calib_done_sent = False

    try:
        while True:
            for p in processes:
                if not p.is_alive():
                    print(f"Orchestrator: ALERTA — {p.__class__.__name__} terminó.")

            # Detectar quando o modelo está pronto
            if model_ready_flag.value and not calib_done_sent:
                calib_done_sent = True
                print("Orchestrator: Modelo listo → notificando UI")
                ui_signal_queue.put(('MODEL_READY', None))

            # Comandos da UI
            while not ui_cmd.empty():
                try:
                    cmd, params = ui_cmd.get_nowait()
                    _route(cmd, params, engine_cmd, hw_sync)
                    # Se inicia nova calibração, resetar bandeira
                    if cmd == 'START_CALIB':
                        calib_done_sent = False
                        model_ready_flag.value = False
                except Exception:
                    pass

            time.sleep(0.02)

    except KeyboardInterrupt:
        shutdown()


def _route(cmd, params, engine_cmd, hw_sync):
    if cmd == 'START_CALIB':
        engine_cmd.put(('START_CALIB', None))
        print("Orchestrator: Fase → CALIBRAÇÃO iniciada")
    elif cmd == 'CALIB_POINT':
        engine_cmd.put(('CALIB_POINT', params))
        print(f"Orchestrator: Ponto de calibração → {params}")
    elif cmd == 'STOP_CALIB':
        engine_cmd.put(('STOP_CALIB', None))
        print("Orchestrator: Fase → Treinando modelo...")
    elif cmd == 'START_SEQUENCE':
        engine_cmd.put(('START_SEQUENCE', params))
        print(f"Orchestrator: Secuencia '{params[0]}' punto {params[1]}")
    elif cmd == 'IDLE':
        engine_cmd.put(('IDLE', None))
    elif cmd == 'GENERATE_REPORT':
        engine_cmd.put(('GENERATE_REPORT', None))
        print("Orchestrator: Fase → Gerando relatório final...")
    elif cmd == 'ADJUST_SYNC':
        hw_sync.put(params)
    elif cmd == 'QUIT':
        raise KeyboardInterrupt


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()