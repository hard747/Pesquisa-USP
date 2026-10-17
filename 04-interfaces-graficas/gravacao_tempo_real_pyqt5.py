import cv2
import time
from pyudev import Context
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import sys

class VideoRecorderThread(QThread):
    frameCaptured = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.dispositivo = obtener_dispositivo_por_identificador('1415', '2000')
        self.recording = False

    def run(self):
        cap = cv2.VideoCapture(self.dispositivo)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        width, height = 640, 480
        out_color = cv2.VideoWriter('video_salida.mp4', fourcc, fps, (width, height))
        out_bw = cv2.VideoWriter('video_salida_bw.mp4', fourcc, fps, (width, height), isColor=False)

        while self.recording:  # El bucle seguirá hasta que se detenga manualmente
            ret, frame = cap.read()

            if not ret:
                print("No se pudo capturar el frame")
                break

            # Guardar el frame en el video a color
            out_color.write(frame)

            # Convertir a escala de grises y guardar en el video blanco y negro
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_bw.write(gray_frame)

            qImg = self.convert_frame_to_qimage(frame)
            self.frameCaptured.emit(qImg)

        cap.release()
        out_color.release()
        out_bw.release()

    def startRecording(self):
        self.recording = True
        self.start()

    def stopRecording(self):
        self.recording = False
        self.wait()

    def convert_frame_to_qimage(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(frame_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

def obtener_dispositivo_por_identificador(id_vendor, id_model):
    ctx = Context()
    for device in ctx.list_devices(subsystem='video4linux'):
        properties = device.properties
        if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
            if properties['ID_VENDOR_ID'] == id_vendor or properties['ID_MODEL_ID'] == id_model:
                return device.device_node

class VideoRecorderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.recordingThread = None

    def initUI(self):
        self.setWindowTitle("Video Recorder")
        self.setGeometry(100, 100, 640, 480)

        self.label_video = QLabel(self)
        self.label_video.setAlignment(Qt.AlignCenter)

        self.button_grabar = QPushButton("Grabar")
        self.button_grabar.clicked.connect(self.iniciar_grabacion)

        self.button_detener = QPushButton("Detener")
        self.button_detener.setEnabled(False)
        self.button_detener.clicked.connect(self.detener_grabacion)

        self.label_tiempo = QLabel(self)
        self.label_tiempo.setAlignment(Qt.AlignCenter)
        self.label_tiempo.setText("Tiempo: 0 seg")

        layout = QVBoxLayout()
        layout.addWidget(self.label_video)
        layout.addWidget(self.label_tiempo)
        layout.addWidget(self.button_grabar)
        layout.addWidget(self.button_detener)

        self.setLayout(layout)

    def iniciar_grabacion(self):
        self.recordingThread = VideoRecorderThread()
        self.recordingThread.frameCaptured.connect(self.mostrar_frame)
        self.recordingThread.startRecording()
        self.button_grabar.setEnabled(False)
        self.button_detener.setEnabled(True)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_tiempo)
        self.tiempo_inicio = time.time()
        self.timer.start(1000)  # Actualizar cada segundo

    def detener_grabacion(self):
        self.recordingThread.stopRecording()
        self.button_grabar.setEnabled(True)
        self.button_detener.setEnabled(False)

        self.timer.stop()

    def mostrar_frame(self, qImage):
        pixmap = QPixmap.fromImage(qImage)
        self.label_video.setPixmap(pixmap)

    def actualizar_tiempo(self):
        tiempo_actual = time.time()
        tiempo_transcurrido = int(tiempo_actual - self.tiempo_inicio)
        self.label_tiempo.setText(f"Tiempo: {tiempo_transcurrido} seg")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoRecorderApp()
    window.show()
    sys.exit(app.exec_())
