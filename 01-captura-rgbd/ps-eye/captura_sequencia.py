import cv2

from pyudev import Context


def obtener_dispositivo_por_identificador(id_vendor, id_model):
    ctx = Context()
    for device in ctx.list_devices(subsystem='video4linux'):
        properties = device.properties
        if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
            if properties['ID_VENDOR_ID'] == id_vendor or properties['ID_MODEL_ID'] == id_model:
                return device.device_node


def capturar_fotogramas(num_frames):

    dispositivo= obtener_dispositivo_por_identificador('1415','2000')
    cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("No se pudo abrir la cámara PlayStation")
        return

    for i in range(num_frames):
        # Lee un frame de la cámara
        ret, frame = cap.read()

        if not ret:
            print(f"No se pudo capturar el frame {i+1}")
            break

        # Guarda el frame como una imagen
        cv2.imwrite(f"fotograma_{i+1}.png", frame)

    # Libera la cámara
    cap.release()

if __name__ == "__main__":
    num_frames = 50  # Define el número de fotogramas que deseas capturar
    capturar_fotogramas(num_frames)
