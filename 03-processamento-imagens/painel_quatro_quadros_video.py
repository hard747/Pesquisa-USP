import cv2
import numpy as np
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def process_depth_canny(depth_image):
    return cv2.Canny(depth_image, 100, 200)

def process_rgb_orb(rgb_image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(rgb_image, None)
    return cv2.drawKeypoints(rgb_image, kp, None, color=(0,255,0), flags=0)

def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # Configura el codec y el VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dividir el marco en regiones
        h, w, _ = frame.shape
        top_left = frame[:h//2, :w//2]
        top_right = frame[:h//2, w//2:]
        bottom_left = frame[h//2:, :w//2]
        bottom_right = frame[h//2:, w//2:]

        # Convertir la parte superior derecha a escala de grises
        top_right_gray = cv2.cvtColor(top_right, cv2.COLOR_BGR2GRAY)

        # Aplicar el algoritmo Canny en la parte inferior izquierda
        bottom_left_canny = process_depth_canny(bottom_left)

        # Procesar la parte inferior derecha con ORB
        bottom_right_orb = process_rgb_orb(bottom_right)

        # Combinar las regiones procesadas
        combined_top = np.hstack((top_left, cv2.cvtColor(top_right_gray, cv2.COLOR_GRAY2BGR)))
        combined_bottom = np.hstack((cv2.cvtColor(bottom_left_canny, cv2.COLOR_GRAY2BGR), bottom_right_orb))
        final_image = np.vstack((combined_top, combined_bottom))

        # Escribir el fotograma procesado en el archivo de salida
        out.write(final_image)

    cap.release()
    out.release()

if __name__ == '__main__':
    main(str(DADOS / 'videos' / 'video_rgb.mp4'), 'video_procesado.mp4')
