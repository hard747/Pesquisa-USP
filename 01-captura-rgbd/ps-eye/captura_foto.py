import cv2
from pyudev import Context
import sys

# if len(sys.argv) != 3:
#     print("Se requieren dos argumentos da camera: id_vendor y id_product")
#     sys.exit(1)
#
# id_vendor = sys.argv[1]
# id_model = sys.argv[2]

def obtener_dispositivo_por_identificador(id_vendor,id_model):
    ctx = Context()
    for device in ctx.list_devices(subsystem='video4linux'):
        properties = device.properties #otra opcion que reemplaza device por propierties
        if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
            if properties['ID_VENDOR_ID'] == id_vendor or properties['ID_MODEL_ID'] == id_model:
        # if device.get('ID_MODEL_ID') == identificador:
                return device.device_node

def capturar_imagen():
    id_vendor = '1415'  # Parte del identificador 1415:2000
    id_model = '2000'  # Parte del identificador 1415:2000
    dispositivo = obtener_dispositivo_por_identificador(id_vendor,id_model)

    # dispositivo = True
    if dispositivo:
        # Intenta abrir la cámara con el identificador específico
        cap = cv2.VideoCapture(dispositivo)
        # cap = cv2.VideoCapture(4)

        if not cap.isOpened():
            print(f"No se pudo abrir el dispositivo")
            return

        # Lee un frame de la cámara
        ret, frame = cap.read()

        if not ret:
            print("No se pudo capturar el frame")
            return

        # Guarda el frame como una imagen
        cv2.imwrite("captura.png", frame)

        # Libera la cámara
        cap.release()
    else:
        print(f"No se encontró ningún dispositivo con el identificador")

if __name__ == "__main__":
    capturar_imagen()
