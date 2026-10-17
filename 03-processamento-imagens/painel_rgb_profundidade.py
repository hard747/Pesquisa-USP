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

def main(rgb_image_path, depth_image_path, output_image_path):
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    depth_canny = process_depth_canny(depth_image)
    rgb_orb = process_rgb_orb(rgb_image)

    # Convertir depth_canny a una imagen a color
    depth_canny_color = cv2.cvtColor(depth_canny, cv2.COLOR_GRAY2BGR)

    # Crear paneles
    top_row = np.hstack((rgb_image, depth_image))
    bottom_row = np.hstack((rgb_orb, depth_canny_color))
    final_image = np.vstack((top_row, bottom_row))

    # Guardar la imagen
    cv2.imwrite(output_image_path, final_image)

if __name__ == '__main__':
    main(str(DADOS / 'exemplos' / 'rgb.jpg'), str(DADOS / 'exemplos' / 'profundidade.jpg'), 'output_image.jpg')
