import cv2
from pyudev import Context
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
import sys


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Captura de Imagen")
        self.setGeometry(100, 100, 400, 200)

        self.label_vendor = QLabel("ID Vendor:")
        self.entry_vendor = QLineEdit()

        self.label_model = QLabel("ID Model:")
        self.entry_model = QLineEdit()

        self.button_capturar = QPushButton("Capturar Imagen")
        self.button_capturar.clicked.connect(self.capturar_desde_gui)

        self.label_imagen = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.label_vendor)
        layout.addWidget(self.entry_vendor)
        layout.addWidget(self.label_model)
        layout.addWidget(self.entry_model)
        layout.addWidget(self.button_capturar)
        layout.addWidget(self.label_imagen)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def capturar_desde_gui(self):
        id_vendor = self.entry_vendor.text()
        id_model = self.entry_model.text()
        imagen_capturada = capturar_imagen(id_vendor, id_model)
        if imagen_capturada:
            image = QImage(imagen_capturada)
            pixmap = QPixmap.fromImage(image)
            self.label_imagen.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
