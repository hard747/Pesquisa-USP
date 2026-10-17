import cv2
import time
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import sys
import numpy as np

def process_depth_canny(depth_frame):
    return cv2.Canny(depth_frame, 100, 200)

def process_rgb_orb(rgb_frame):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(rgb_frame, None)
    return cv2.drawKeypoints(rgb_frame, kp, None, color=(0,255,0), flags=0)

def convert_frame_to_qimage(frame):
    height, width, channel = frame.shape
    bytesPerLine = 3 * width
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(frame_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)

class VideoRecorderThread(QThread):
    frameCaptured = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.recording = False

    def run(self):
        cap = cv2.VideoCapture(2)  # Abre la cámara

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        width, height = 640, 480
        out_color = cv2.VideoWriter('video_salida.mp4', fourcc, fps, (width, height))
        out_bw = cv2.VideoWriter('video_salida_bw.mp4', fourcc, fps, (width, height), isColor=False)

        while self.recording:
            ret, frame = cap.read()

            if not ret:
                print("No se pudo capturar el frame")
                break

            # Guardar el frame en el video a color
            out_color.write(frame)

            # Convertir a escala de grises y guardar en el video blanco y negro
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out_bw.write(gray_frame)

            qImg = convert_frame_to_qimage(frame)
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

class VideoProcessorThread(QThread):
    frameProcessed = pyqtSignal(QImage)

    def __init__(self, rgb_video_path, depth_video_path):
        super().__init__()
        self.rgb_video_path = rgb_video_path
        self.depth_video_path = depth_video_path

    def run(self):
        cap_rgb = cv2.VideoCapture(self.rgb_video_path)
        cap_depth = cv2.VideoCapture(self.depth_video_path)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        while(cap_rgb.isOpened() and cap_depth.isOpened()):
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_depth, frame_depth = cap_depth.read()

            if not ret_rgb or not ret_depth:
                break

            depth_canny = process_depth_canny(frame_depth)
            rgb_orb = process_rgb_orb(frame_rgb)

            depth_canny_color = cv2.cvtColor(depth_canny, cv2.COLOR_GRAY2BGR)

            top_row = np.hstack((frame_rgb, frame_depth))
            bottom_row = np.hstack((rgb_orb, depth_canny_color))
            final_frame = np.vstack((top_row, bottom_row))

            qImg = convert_frame_to_qimage(final_frame)
            self.frameProcessed.emit(qImg)

        cap_rgb.release()
        cap_depth.release()

class MainApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        self.recordingThread = None
        self.processingThread = None

    def initUI(self):
        self.setWindowTitle("Video Processor")
        self.setGeometry(100, 100, 640, 480)

        self.label_video = QLabel(self)
        self.label_video.setAlignment(Qt.AlignCenter)

        self.button_grabar = QPushButton("Grabar")
        self.button_grabar.clicked.connect(self.toggle_grabar)

        self.button_procesar = QPushButton("Procesar Videos")
        self.button_procesar.setEnabled(False)
        self.button_procesar.clicked.connect(self.procesar_videos)

        self.label_tiempo = QLabel(self)
        self.label_tiempo.setAlignment(Qt.AlignCenter)
        self.label_tiempo.hide()

        layout = QVBoxLayout()
        layout.addWidget(self.label_video)
        layout.addWidget(self.button_grabar)
        layout.addWidget(self.button_procesar)
        layout.addWidget(self.label_tiempo)

        self.setLayout(layout)

    def toggle_grabar(self):
        if not self.recordingThread or not self.recordingThread.isRunning():
            self.iniciar_grabacion()
        else:
            self.detener_grabacion()

    def iniciar_grabacion(self):
        self.recordingThread = VideoRecorderThread()
        self.recordingThread.frameCaptured.connect(self.mostrar_frame)
        self.recordingThread.startRecording()
        self.button_procesar.setEnabled(True)
        self.button_grabar.setText("Detener")
        self.label_tiempo.show()

        self.tiempo_inicio = time.time()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.actualizar_tiempo)
        self.timer.start(1000)  # Actualizar cada segundo

    def detener_grabacion(self):
        self.recordingThread.stopRecording()
        self.button_grabar.setText("Grabar")
        self.button_procesar.setEnabled(True)
        self.label_tiempo.hide()

        if hasattr(self, 'timer'):
            self.timer.stop()

    def mostrar_frame(self, qImage):
        pixmap = QPixmap.fromImage(qImage)
        self.label_video.setPixmap(pixmap)

    def actualizar_tiempo(self):
        tiempo_actual = time.time()
        tiempo_transcurrido = int(tiempo_actual - self.tiempo_inicio)
        self.label_tiempo.setText(f"Tiempo: {tiempo_transcurrido} seg")

    def procesar_videos(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        rgb_video_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video RGB", "", "Videos (*.mp4 *.avi);;All Files (*)", options=options)

        depth_video_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video de Profundidad", "", "Videos (*.mp4 *.avi);;All Files (*)", options=options)

        if rgb_video_path and depth_video_path:
            self.processingThread = VideoProcessorThread(rgb_video_path, depth_video_path)
            self.processingThread.frameProcessed.connect(self.mostrar_frame)
            self.processingThread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
