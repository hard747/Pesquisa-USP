import cv2
import threading

def capturar_y_guardar(identificador, nombre_archivo):
    # Intenta abrir la cámara con el identificador especificado
    cap = cv2.VideoCapture(identificador)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con el identificador {identificador}")
        return

    # Lee un frame de la cámara
    ret, frame = cap.read()

    if not ret:
        print(f"No se pudo capturar el frame de la cámara {identificador}")
        return

    # Guarda el frame como una imagen
    cv2.imwrite(nombre_archivo, frame)

    # Libera la cámara
    cap.release()

def capturar_y_guardar_con_thread(identificador, nombre_archivo):
    thread = threading.Thread(target=capturar_y_guardar, args=(identificador, nombre_archivo))
    thread.start()

if __name__ == "__main__":
    # Captura y guarda la imagen del identificador 6 en un hilo
   # capturar_y_guardar_con_thread(6, "captura_PS.png")

    # Captura y guarda la imagen del identificador 2 en un hilo
    capturar_y_guardar_con_thread(2, "captura_rgb.png")

    # Captura y guarda la imagen del identificador 4 en un hilo
    capturar_y_guardar_con_thread(4, "captura_depth.png")

print("salida satisfactoria")
