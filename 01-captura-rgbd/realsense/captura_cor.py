import pyrealsense2 as rs
import numpy as np
import cv2

# Configuración del pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

# Inicia el pipeline
pipeline.start(config)

try:
    while True:
        # Espera a que lleguen los datos del frame
        frames = pipeline.wait_for_frames()

        # Obtén el frame de color
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convierte el frame a una matriz de NumPy
        color_image = np.asanyarray(color_frame.get_data())

        # Muestra la imagen en una ventana
        cv2.imshow('Color Frame', color_image)

        # Espera a una tecla y verifica si es la tecla 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cierra el pipeline al salir
    pipeline.stop()
    cv2.destroyAllWindows()
