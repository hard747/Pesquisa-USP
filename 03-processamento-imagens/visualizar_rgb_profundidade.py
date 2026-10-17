import cv2
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def mostrar_imagenes_separadas(rgb_path, depth_path):
    # Leer la imagen RGB y la imagen de profundidad
    rgb_image = cv2.imread(rgb_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Mostrar la imagen RGB
    cv2.imshow('Imagen RGB', rgb_image)
    cv2.waitKey(0)

    # Mostrar la imagen de profundidad (en escala de grises para visualización)
    cv2.imshow('Imagen de Profundidad', cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(0)

    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

# Llamamos a la función con las rutas de las imágenes
mostrar_imagenes_separadas(str(DADOS / 'exemplos' / 'rgb.jpg'), str(DADOS / 'exemplos' / 'profundidade.jpg'))
