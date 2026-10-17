import cv2
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def convertir_a_gris(video_entrada, video_salida):
    cap = cv2.VideoCapture(video_entrada)

    # Obtiene el ancho y alto del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Configura el codec para el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_salida, fourcc, 30.0, (width, height), isColor=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convierte el frame a escala de grises
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Escribe el frame en el video de salida
        out.write(frame_gris)

    # Libera los recursos
    cap.release()
    out.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    video_entrada = str(DADOS / 'videos' / 'video_rgb.mp4')
    video_salida = 'video_gris.mp4'
    convertir_a_gris(video_entrada, video_salida)
