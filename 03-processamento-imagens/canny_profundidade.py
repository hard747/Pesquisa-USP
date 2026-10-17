import cv2
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def process_depth_canny(depth_image_path):
    # Leer la imagen de profundidad
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    # Aplicar Canny
    depth_canny = cv2.Canny(depth_image, 100, 200)

    # Guardar la imagen resultante
    cv2.imwrite('depth_canny.jpg', depth_canny)

if __name__ == '__main__':
    process_depth_canny(str(DADOS / 'exemplos' / 'profundidade.jpg'))
