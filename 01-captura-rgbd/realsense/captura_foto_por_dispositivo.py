import pyrealsense2 as rs
import numpy as np
import cv2

def configurar_camara(identificador):
    # Configura el pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(identificador)
    pipeline.start(config)
    return pipeline

def capturar_imagen(pipeline):
    # Inicia el pipeline
    frames = pipeline.wait_for_frames()

    # Obtiene el frame de color
    color_frame = frames.get_color_frame()

    if color_frame:
        # Convierte el frame a una matriz de numpy
        frame = np.asanyarray(color_frame.get_data())

        return frame

if __name__ == "__main__":
    # Configura la cámara Intel RealSense
    pipeline = configurar_camara('041e:4099')  # Reemplaza con el identificador correcto si es necesario

    # Captura la imagen
    frame = capturar_imagen(pipeline)

    if frame is not None:
        # Guarda el frame como una imagen
        cv2.imwrite("captura.png", frame)

    # Detiene el pipeline
    pipeline.stop()
