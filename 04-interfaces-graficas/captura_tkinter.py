import cv2
from pyudev import Context
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def obtener_dispositivo_por_identificador(id_vendor, id_model):
    ctx = Context()
    for device in ctx.list_devices(subsystem='video4linux'):
        properties = device.properties
        if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
            if properties['ID_VENDOR_ID'] == id_vendor or properties['ID_MODEL_ID'] == id_model:
                return device.device_node


def capturar_imagen(id_vendor, id_model):
    dispositivo = obtener_dispositivo_por_identificador(id_vendor, id_model)

    if dispositivo:
        cap = cv2.VideoCapture(dispositivo)
        if not cap.isOpened():
            print(f"No se pudo abrir el dispositivo")
            return
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame")
            return
        cv2.imwrite("captura.png", frame)
        cap.release()
        return "captura.png"
    else:
        print(f"No se encontró ningún dispositivo con el identificador")


def capturar_desde_gui():
    id_vendor = entry_vendor.get()
    id_model = entry_model.get()
    imagen_capturada = capturar_imagen(id_vendor, id_model)
    if imagen_capturada:
        image = Image.open(imagen_capturada)
        photo = ImageTk.PhotoImage(image)
        label_imagen.config(image=photo)
        label_imagen.image = photo


# Crear la ventana tkinter
root = tk.Tk()

# Crear Entry widgets para el id_vendor y id_model
label_vendor = ttk.Label(root, text="ID Vendor:")
label_vendor.grid(row=0, column=0)
entry_vendor = ttk.Entry(root)
entry_vendor.grid(row=0, column=1)

label_model = ttk.Label(root, text="ID Model:")
label_model.grid(row=1, column=0)
entry_model = ttk.Entry(root)
entry_model.grid(row=1, column=1)

# Botón para capturar la imagen
button_capturar = ttk.Button(root, text="Capturar Imagen", command=capturar_desde_gui)
button_capturar.grid(row=2, columnspan=2)

# Label para mostrar la imagen capturada
label_imagen = ttk.Label(root)
label_imagen.grid(row=3, columnspan=2)

# Iniciar el bucle principal de tkinter
root.mainloop()
