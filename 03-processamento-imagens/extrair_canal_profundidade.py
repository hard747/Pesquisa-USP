import sys

import open3d as o3d
import numpy as np

def extraer_canal_depth(imagen_rgbd):
    # Cargar la imagen RGBD
    rgbd_image = o3d.io.read_image(imagen_rgbd)

    # Convertir a numpy array
    depth_array = np.array(rgbd_image)

    # Extraer el canal de profundidad
    depth_channel = depth_array[:, :, 3]

    return depth_channel

if __name__ == "__main__":
    # Ruta de la imagen RGBD (reemplaza con la tuya)
    imagen_rgbd = sys.argv[1]

    # Extraer el canal de profundidad
    depth_channel = extraer_canal_depth(imagen_rgbd)

    # Guardar el canal de profundidad como una nueva imagen (opcional)
    o3d.io.write_image("depth_channel.png", o3d.Image(depth_channel))
