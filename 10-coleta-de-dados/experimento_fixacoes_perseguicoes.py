import cv2
import numpy as np
import time
import tkinter as tk
import random
import os
import datetime
import csv
from PIL import Image, ImageTk

try:
    import pyudev
    PYUDEV_DISPONIBLE = True
except ImportError:
    print("ADVERTENCIA: 'pyudev' no está instalado (pip install pyudev).")
    PYUDEV_DISPONIBLE = False

# --------------------------------------
# CONFIGURAÇÃO GERAL
# --------------------------------------

WINDOW_NAME = "Experimento de Coleta (Fix + Per)"
DEBUG_DRAW_RHOMBUS = False # Desenhar los contornos de rombo para depuração

# --- Configuração da Nomenclatura ---
MAIN_DATA_FOLDER = "data"

# --- Um único ID de Sujeito/Sessão ---
# Mude esta linha para a sessão que você quer correr (ex: "P10", "P16", "P17")
SUBJECT_ID = "P18"
# ------------------------------------------

EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
X_COORDS = [135, 410, 685, 960, 1235, 1510, 1785]
Y_COORDS = [108, 324, 540, 756, 972]

# Cores (BGR)
CROSS_COLOR = (255, 255, 255) 
TEXT_COLOR = (255, 255, 255) 
DISTRACTOR_COLOR = (0, 0, 255) 
DEBUG_LINE_COLOR = (50, 50, 50) 
CROSS_SIZE = 30

# --- CABEÇALHO CSV (Único para ambos) ---
CSV_HEADER = [
    'timestamp', 'file_name', 'status', 
    'pupil_x', 'pupil_y', 'glint_x', 'glint_y', 
    'white', 
    'tar1_x', 'tar1_y', 'tar2_x', 'tar2_y', 'tar3_x', 'tar3_y', 
    'tar4_x', 'tar4_y', 'tar5_x', 'tar5_y', 'tar6_x', 'tar6_y'
]

# --------------------------------------
# CONFIGURAÇÃO FIXAÇÕES (14 Provas)
# --------------------------------------
FIX_CENTER_TIME = 1.0
FIX_GROUP_TIME = 2.0
FIX_TOTAL_TIME = FIX_CENTER_TIME + (FIX_GROUP_TIME * 5) + FIX_CENTER_TIME # Total 12s
FIXACOES_TARGET_GROUPS = {
    "Canto se": [9, 8, 1, 2, 3], "Canto sd": [13, 14, 7, 6, 5], "Anc11": [11, 12, 4, 10, 16],
    "Anc23": [23, 22, 30, 24, 16], "Anc27": [27, 20, 26, 28, 34], "W": [15, 31, 25, 33, 21],
    "M": [29, 17, 32, 19, 35]
}
FIXACOES_GROUP_TO_CODE = {
    "Canto se": "p", "Canto sd": "q", "Anc11": "r", "Anc23": "s",
    "Anc27": "t", "W": "u", "M": "v"
}

# --------------------------------------
# CONFIGURAÇÃO PERSEGUIÇÕES (18 Provas)
# --------------------------------------
PER_STATIC_TIME = 1.0
PER_MOTION_TIME = 5.0
PERSEGUICOES_TARGET_GROUPS = {
    "A9_Balao": [2, 10, 16, 8], "A11_Balao": [4, 12, 18, 10], "A13_Balao": [6, 14, 20, 12],
    "A23_Balao": [16, 24, 30, 22], "A25_Balao": [18, 26, 32, 24], "A27_Balao": [20, 28, 34, 26],
}
PERSEGUICOES_GROUP_TO_CODE = {
    "A9_Balao": "a", "A11_Balao": "b", "A13_Balao": "c",
    "A23_Balao": "d", "A25_Balao": "e", "A27_Balao": "f"
}
VELOCIDADES = [
    {"label": "rapido", "time": PER_MOTION_TIME, "rots": 3} 
]
SENTIDOS = ["horario", "antihorario"]


# --------------------------------------
# CLASE MANEJADOR DE CÁMARA
# --------------------------------------
class CameraManager:
    PS3_EYE_VENDOR_ID = '1415'
    PS3_EYE_MODEL_ID = '2000'
    def __init__(self):
        self.captura = None
        self.width = 0
        self.height = 0
        self.fps = 0
    def obter_id_camera_ps3_eye(self):
        if not PYUDEV_DISPONIBLE: return 0
        try:
            contexto = pyudev.Context()
            for dispositivo in contexto.list_devices(subsystem='video4linux'):
                props = dispositivo.properties
                if 'ID_VENDOR_ID' in props and 'ID_MODEL_ID' in props:
                    if props['ID_VENDOR_ID'] == self.PS3_EYE_VENDOR_ID and props['ID_MODEL_ID'] == self.PS3_EYE_MODEL_ID:
                        try: return int(dispositivo.device_node.replace('/dev/video', ''))
                        except: return 0
            return 0
        except: return 0
    def iniciar_camera(self):
        id_camera = self.obter_id_camera_ps3_eye()
        self.captura = cv2.VideoCapture(id_camera)
        if not self.captura.isOpened():
            print(f"Erro: No se pudo abrir cámara ID {id_camera}")
            return False
        print("Intentando configurar 640x480 @ 60 FPS...")
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.captura.set(cv2.CAP_PROP_FPS, 60)
        time.sleep(0.5)
        self.width = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.captura.get(cv2.CAP_PROP_FPS)
        print(f"Resolución REAL de la cámara: {self.width}x{self.height}")
        if self.fps == 0: self.fps = 60
        return True
    def liberar_camera(self):
        if self.captura:
            self.captura.release()
            print("Cámara liberada.")

# --------------------------------------
# CLASE DISPLAYAPP (DE TKINTER)
# --------------------------------------
class DisplayApp:
    def __init__(self, width, height):
        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)
        self.root.attributes('-fullscreen', True)
        self.root.config(cursor="none")
        self.width = width
        self.height = height
        self.label = tk.Label(self.root, borderwidth=0, bg="black")
        self.label.pack(expand=True, fill=tk.BOTH)
        self.last_key_pressed = None
        self.root.bind('<Key>', self._key_press_handler)
        self.root.bind('<Escape>', lambda e: self._key_press_handler(e, 'Escape'))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_frame(np.zeros((height, width, 3), dtype=np.uint8))
        self.update()
    def _key_press_handler(self, event, override_keysym=None):
        if override_keysym: self.last_key_pressed = override_keysym
        else: self.last_key_pressed = event.keysym
    def get_key(self):
        key = self.last_key_pressed
        self.last_key_pressed = None 
        return key
    def show_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.label.config(image=img_tk)
        self.label.image = img_tk
    def update(self):
        self.root.update_idletasks()
        self.root.update()
    def close(self):
        print("Cerrando ventana (Tkinter).")
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error al cerrar: {e}")

# --------------------------------------
# FUNCIONES AUXILIARES COMUNS
# --------------------------------------
def get_screen_resolution():
    try:
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except:
        return EXPECTED_WIDTH, EXPECTED_HEIGHT

def get_grid_centers_from_pixels(screen_width, screen_height):
    scale_x = screen_width / EXPECTED_WIDTH
    scale_y = screen_height / EXPECTED_HEIGHT
    centers = []
    for y in Y_COORDS:
        for x in X_COORDS:
            centers.append((int(x * scale_x), int(y * scale_y)))
    return centers

def draw_cross(image, center, color, size, thickness=2):
    cx, cy = int(center[0]), int(center[1])
    half = size//2
    cv2.line(image, (cx-half, cy), (cx+half, cy), color, thickness)
    cv2.line(image, (cx, cy-half), (cx, cy+half), color, thickness)

def show_message_and_wait(app, screen_width, screen_height, message):
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(img, message, ((screen_width-tw)//2, (screen_height+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 2)
    app.show_frame(img)
    start_time = time.time()
    GRACE_PERIOD_SEC = 0.5 
    while True:
        app.update()
        key = app.get_key()
        if key == 'space': 
            return True
        elapsed_time = time.time() - start_time
        if elapsed_time > GRACE_PERIOD_SEC:
            if key == 'q' or key == 'Escape':
                print("Salida manual detectada.")
                return False

def show_saving_message(app, screen_width, screen_height):
    img = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    message = "Salvando... Por favor, aguarde."
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.putText(img, message, ((screen_width-tw)//2, (screen_height+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 2)
    app.show_frame(img)
    app.update()

def measure_camera_fps(capture, measure_seconds=1.0):
    if not capture or not capture.isOpened(): return 0
    start = time.time()
    frames = 0
    while time.time() - start < measure_seconds:
        ret = capture.grab()
        if not ret:
            ret2,_ = capture.read()
            if not ret2: break
        frames += 1
    elapsed = time.time() - start
    if elapsed == 0: return 0
    return frames / elapsed

# -------------------------------------------------------------
# --- FUNÇÕES DE LÓGICA DE FUNCIONAMENTO (COM BUFFER E OTIMIZADAS) ---
# -------------------------------------------------------------

# --- LÓGICA DA FIXAÇÃO (12s) ---
def run_fixacao_trial(app, cam, screen_width, screen_height, validation_centers, trial_info):
    
    local_trial_buffer = [] 
    target_ids = FIXACOES_TARGET_GROUPS[trial_info['group_name']]
    center_coords = validation_centers[18-1]
    start_time = time.time()
    coords = (0, 0)
    frame_counter = 0
    
    while True:
        frame_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= FIX_TOTAL_TIME:
            break 

        # Otimização (Só desenha 1 a cada 2 frames)
        if frame_counter % 2 == 0:
            img_estimulo = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            
            if elapsed < FIX_CENTER_TIME:
                coords = center_coords
            elif elapsed < FIX_CENTER_TIME + FIX_GROUP_TIME*1:
                coords = validation_centers[target_ids[0]-1]
            elif elapsed < FIX_CENTER_TIME + FIX_GROUP_TIME*2:
                coords = validation_centers[target_ids[1]-1]
            elif elapsed < FIX_CENTER_TIME + FIX_GROUP_TIME*3:
                coords = validation_centers[target_ids[2]-1]
            elif elapsed < FIX_CENTER_TIME + FIX_GROUP_TIME*4:
                coords = validation_centers[target_ids[3]-1]
            elif elapsed < FIX_CENTER_TIME + FIX_GROUP_TIME*5:
                coords = validation_centers[target_ids[4]-1]
            else:
                coords = center_coords
                
            draw_cross(img_estimulo, coords, CROSS_COLOR, CROSS_SIZE)
            app.show_frame(img_estimulo)

        # Captura da Câmera (acontece em TODOS os frames)
        ret, frame = cam.captura.read()
        if ret:
            current_timestamp = time.time() 
            csv_row = [
                current_timestamp, None, 1, -1, -1, -1, -1, 1,
                coords[0], coords[1],
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
            ]
            local_trial_buffer.append( (frame.copy(), csv_row) )

        app.update()
        key = app.get_key()
        if key == 'q' or key == 'Escape':
            print("Salida manual detectada.")
            return None, False 

    return local_trial_buffer, True

# --- FUNÇÕES DE PERSEGUIÇÃO (Estático) ---
def run_perseguicao_static(
    app, screen_width, screen_height, validation_centers, all_paths_rotated, 
    target_index_num, duration_sec, 
    cam
):
    local_trial_buffer = [] 
    all_midpoints = []
    all_vertex_sets_int = [] 
    for path_ids in all_paths_rotated:
        coords = [validation_centers[pid - 1] for pid in path_ids]
        all_vertex_sets_int.append([(int(c[0]), int(c[1])) for c in coords])
        coords_start = coords[0] 
        coords_end = coords[1]
        mid_x = coords_start[0] * 0.5 + coords_end[0] * 0.5
        mid_y = coords_start[1] * 0.5 + coords_end[1] * 0.5
        all_midpoints.append((mid_x, mid_y))

    flat_target_coords = [coord for pair in all_midpoints for coord in pair]
    start_time = time.time()
    frame_counter = 0
    
    while time.time() - start_time < duration_sec:
        frame_counter += 1
        
        if frame_counter % 2 == 0:
            img_alvo = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            for i in range(len(all_paths_rotated)):
                if DEBUG_DRAW_RHOMBUS:
                    p = all_vertex_sets_int[i]
                    cv2.line(img_alvo, p[0], p[1], DEBUG_LINE_COLOR, 1)
                    cv2.line(img_alvo, p[1], p[2], DEBUG_LINE_COLOR, 1)
                    cv2.line(img_alvo, p[2], p[3], DEBUG_LINE_COLOR, 1)
                    cv2.line(img_alvo, p[3], p[0], DEBUG_LINE_COLOR, 1)
                midpoint_coords = all_midpoints[i]
                color = CROSS_COLOR if i == target_index_num else DISTRACTOR_COLOR
                draw_cross(img_alvo, midpoint_coords, color, CROSS_SIZE)
            app.show_frame(img_alvo)
        
        ret, frame = cam.captura.read()
        if ret:
            current_timestamp = time.time()
            csv_row = [
                current_timestamp, None, 1, -1, -1, -1, -1,
                target_index_num + 1,
            ] + flat_target_coords 
            local_trial_buffer.append( (frame.copy(), csv_row) )
        
        app.update()
        key = app.get_key()
        if key == 'q' or key == 'Escape':
            print("Salida manual detectada.")
            return None, False 
    return local_trial_buffer, True

# --- FUNÇÕES DE PERSEGUIÇÃO (Movimento) ---
def run_perseguicao_motion(
    app, screen_width, screen_height, validation_centers, all_paths_rotated, 
    target_index_num, rotation_time, num_rotations,
    cam
):
    local_trial_buffer = [] 
    POINTS_PER_SHAPE = len(all_paths_rotated[0])
    
    if num_rotations == 0 or POINTS_PER_SHAPE == 0: TIME_PER_SEGMENT = 0.1
    else: TIME_PER_SEGMENT = rotation_time / (num_rotations * POINTS_PER_SHAPE)
    if TIME_PER_SEGMENT == 0: TIME_PER_SEGMENT = 0.001 
    TIME_OFFSET = TIME_PER_SEGMENT / 2.0
    
    print(f"     Iniciando Rotação Contínua ({num_rotations}r em {rotation_time}s)...")
    
    start_time = time.time()
    frame_counter = 0
    
    while True:
        frame_counter += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= rotation_time: break
        
        effective_time = elapsed_time + TIME_OFFSET
        current_segment_total_index = int(effective_time / TIME_PER_SEGMENT)
        time_in_current_segment = effective_time - (current_segment_total_index * TIME_PER_SEGMENT)
        t = time_in_current_segment / TIME_PER_SEGMENT
        t = max(0.0, min(1.0, t))
        
        current_target_coords_list = []
        for i, path_ids in enumerate(all_paths_rotated):
            path_index_start = current_segment_total_index % POINTS_PER_SHAPE 
            path_index_end = (path_index_start + 1) % POINTS_PER_SHAPE 
            point_id_start = path_ids[path_index_start]
            point_id_end = path_ids[path_index_end]
            coords_start = validation_centers[point_id_start - 1]
            coords_end = validation_centers[point_id_end - 1]
            current_x = coords_start[0] * (1 - t) + coords_end[0] * t
            current_y = coords_start[1] * (1 - t) + coords_end[1] * t
            current_target_coords_list.append((current_x, current_y))
        
        if frame_counter % 2 == 0:
            img_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            for i, (current_x, current_y) in enumerate(current_target_coords_list):
                if DEBUG_DRAW_RHOMBUS:
                    pass
                color = CROSS_COLOR if i == target_index_num else DISTRACTOR_COLOR
                draw_cross(img_frame, (current_x, current_y), color, CROSS_SIZE)
            app.show_frame(img_frame)
        
        ret, frame = cam.captura.read()
        if ret:
            current_timestamp = time.time()
            flat_target_coords = [coord for pair in current_target_coords_list for coord in pair]
            csv_row = [
                current_timestamp, None, 1, -1, -1, -1, -1,
                target_index_num + 1,
            ] + flat_target_coords
            local_trial_buffer.append( (frame.copy(), csv_row) )
        
        app.update()
        key = app.get_key()
        if key == 'q' or key == 'Escape':
            print("Salida manual detectada.")
            return None, False 
            
    return local_trial_buffer, True

# --- LÓGICA DA PERSEGUIÇÃO (7s) ---
def run_perseguicao_trial(app, cam, screen_width, screen_height, validation_centers, trial_info):

    target_key_name = trial_info['group_name']
    velocidade = VELOCIDADES[0] 

    all_shape_keys = list(PERSEGUICOES_TARGET_GROUPS.keys())
    potential_distractors = [k for k in all_shape_keys if k != target_key_name]
    distractor_keys = random.sample(potential_distractors, 5)
    trial_shape_keys = [target_key_name] + distractor_keys
    random.shuffle(trial_shape_keys)
    target_index = trial_shape_keys.index(target_key_name)
    
    all_paths_rotated = []
    for key in trial_shape_keys:
        path_ids = PERSEGUICOES_TARGET_GROUPS[key][:]
        sentido = random.choice(SENTIDOS) 
        if sentido == "antihorario": path_ids.reverse() 
        start_index = random.randint(0, len(path_ids) - 1)
        rotated_path = path_ids[start_index:] + path_ids[:start_index]
        all_paths_rotated.append(rotated_path)

    trial_buffer = []
    running = True

    print("     Mostrando Pontos Iniciais (Pontos Médios) por 1.0 seg...")
    buffer1, running = run_perseguicao_static(
        app, screen_width, screen_height, validation_centers, all_paths_rotated, 
        target_index, PER_STATIC_TIME, cam
    )
    if not running: return None, False 
    trial_buffer.extend(buffer1)

    buffer2, running = run_perseguicao_motion( 
        app, screen_width, screen_height, validation_centers, all_paths_rotated, 
        target_index, velocidade["time"], velocidade["rots"], cam
    )
    if not running: return None, False
    trial_buffer.extend(buffer2)
    
    print("     Parando nos Pontos Finais por 1.0 seg...")
    buffer3, running = run_perseguicao_static(
        app, screen_width, screen_height, validation_centers, all_paths_rotated, 
        target_index, PER_STATIC_TIME, cam
    )
    if not running: return None, False
    trial_buffer.extend(buffer3)

    return trial_buffer, True 

# --------------------------------------
# LÓGICA PRINCIPAL (MESTRA)
# --------------------------------------
def main():
    
    cam = CameraManager()
    if not cam.iniciar_camera(): 
        return
    
    SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()
    print(f"Iniciando em resolução fixa: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    
    validation_centers = get_grid_centers_from_pixels(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    measured_fps = measure_camera_fps(cam.captura, 1.0)
    print(f"FPS inicial medido: {measured_fps:.2f} (Isto é apenas informativo)")

    # --- Usar o SUBJECT_ID único ---
    session_path = os.path.join(MAIN_DATA_FOLDER, SUBJECT_ID)
    os.makedirs(session_path, exist_ok=True)
    print(f"Guardando todos os dados em: '{session_path}'")
    # ----------------------------------------

    # --- Criar Lista Mestra de 32 Provas ---
    master_trial_list = []
    
    # 1. Adicionar 14 provas de Fixação
    fixacao_base_list = list(FIXACOES_TARGET_GROUPS.keys()) * 2
    for group_name in fixacao_base_list:
        master_trial_list.append({
            "type": "fixacao",
            "group_name": group_name,
            "group_code": FIXACOES_GROUP_TO_CODE[group_name]
        })
        
    # 2. Adicionar 18 provas de Perseguição
    perseguicao_base_list = list(PERSEGUICOES_TARGET_GROUPS.keys()) * 3
    for group_name in perseguicao_base_list:
         master_trial_list.append({
            "type": "perseguicao",
            "group_name": group_name,
            "group_code": PERSEGUICOES_GROUP_TO_CODE[group_name]
        })
    
    # 3. Aleatorizar a lista mestra
    random.shuffle(master_trial_list)
    total_provas = len(master_trial_list)
    print(f"\nTotal de {total_provas} provas aleatorizadas (14 Fixações, 18 Perseguições).")
    
    app = DisplayApp(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    fixacao_run_counter = {}
    perseguicao_run_counter = {}
    manual_exit = False
    
    try:
        # --- BUCLE MESTRE ---
        for i, trial_info in enumerate(master_trial_list):
            
            print(f"\n--- Iniciando Prova {i+1}/{total_provas} (Tipo: {trial_info['type']}) ---")
            
            msg = f"Prova {i+1}/{total_provas}. Pressione Barra espaco para iniciar."
            if not show_message_and_wait(app, SCREEN_WIDTH, SCREEN_HEIGHT, msg):
                manual_exit = True; break
            
            trial_buffer_data = None
            running = True
            
            if trial_info['type'] == 'fixacao':
                
                group_code = trial_info['group_code']
                current_run = fixacao_run_counter.get(group_code, 0) + 1
                fixacao_run_counter[group_code] = current_run
                safe_name = f"f{group_code}{current_run}"
                
                print(f"     Prova de FIXAÇÃO: {trial_info['group_name']} (Execução {current_run})")
                
                trial_buffer_data, running = run_fixacao_trial(
                    app, cam, SCREEN_WIDTH, SCREEN_HEIGHT, validation_centers, 
                    trial_info
                )

            elif trial_info['type'] == 'perseguicao':
                
                group_code = trial_info['group_code']
                current_run = perseguicao_run_counter.get(group_code, 0) + 1
                perseguicao_run_counter[group_code] = current_run
                safe_name = f"p{group_code}{current_run}"
                
                print(f"     Prova de PERSEGUIÇÃO: {trial_info['group_name']} (Execução {current_run})")
                
                trial_buffer_data, running = run_perseguicao_trial(
                    app, cam, SCREEN_WIDTH, SCREEN_HEIGHT, validation_centers, 
                    trial_info
                )
            
            if not running: manual_exit = True; break

            # --- FASE DE GUARDADO ---
            if trial_buffer_data:
                print(f"     ... Prova {safe_name} completada. Capturados {len(trial_buffer_data)} frames.")
                show_saving_message(app, SCREEN_WIDTH, SCREEN_HEIGHT) # Mostra "Salvando..."
                
                # --- Caminho da pasta usa o 'session_path' unificado ---
                group_folder_path = os.path.join(session_path, safe_name)
                os.makedirs(group_folder_path, exist_ok=True)
                
                csv_data_to_write = []
                
                for frame_index, (frame, csv_row) in enumerate(trial_buffer_data):
                    frame_num_str = f"{frame_index+1:06d}"
                    image_name = f"{safe_name}_{frame_num_str}.jpg" 
                    image_path = os.path.join(group_folder_path, image_name)
                    
                    #Salvar como JPG com 95% de qualidade
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    csv_row[1] = image_name 
                    csv_data_to_write.append(csv_row)
                
                csv_filename = f"{safe_name}_coleta.csv"
                csv_path = os.path.join(group_folder_path, csv_filename)
                try:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(CSV_HEADER) 
                        csv_writer.writerows(csv_data_to_write)
                    print(f"     CSV e {len(trial_buffer_data)} frames salvos exitosamente.")
                except Exception as e:
                    print(f"     ERROR: No se pudo guardar el CSV en {csv_path}. Error: {e}")
            # --- FIM DA FASE DE GUARDADO ---
            
            print(f"--- Prova {i+1}/{total_provas} completada. ---")
            time.sleep(0.5)

        print(f"\n--- TODAS AS {total_provas} PROVAS COMPLETADAS. ---")

    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
    finally:
        print("Cerrando la aplicación.")
        cam.liberar_camera()
        app.close()

if __name__ == "__main__":
    main()