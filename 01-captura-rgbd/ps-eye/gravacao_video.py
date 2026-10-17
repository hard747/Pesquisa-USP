import cv2
import time
from pyudev import Context

def obtener_dispositivo_por_identificador(id_vendor, id_model):
    ctx = Context()
    for device in ctx.list_devices(subsystem='video4linux'):
        properties = device.properties
        if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
            if properties['ID_VENDOR_ID'] == id_vendor or properties['ID_MODEL_ID'] == id_model:
                return device.device_node


def grabar_video(duracion_segundos, nombre_archivo):

    dispositivo= obtener_dispositivo_por_identificador('1415','2000')

    cap = cv2.VideoCapture(dispositivo)

    if not cap.isOpened():
        print("No se pudo abrir la cámara PlayStation")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Cambiado el codec a mp4v
    fps = 30.0
    width, height = 640, 480
    out = cv2.VideoWriter(nombre_archivo, fourcc, fps, (width, height))

    tiempo_inicial = time.time()

    while True:
        tiempo_actual = time.time()
        if tiempo_actual - tiempo_inicial > duracion_segundos:
            break

        ret, frame = cap.read()

        if not ret:
            print("No se pudo capturar el frame")
            break

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    duracion_segundos = 10
    nombre_archivo = "video_salida.mp4"  # Cambiado el nombre a .mp4
    grabar_video(duracion_segundos, nombre_archivo)
