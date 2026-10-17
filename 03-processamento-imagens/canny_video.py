import cv2
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def aplicar_canny(video_entrada, video_salida):
    cap = cv2.VideoCapture(video_entrada)

    # Obtener la información del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definir el codec y crear un objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_salida, fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar Canny al fotograma
        edges = cv2.Canny(frame, 50, 100)

        # Escribir el frame procesado en el archivo de salida
        out.write(edges)

    # Liberar recursos
    cap.release()
    out.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    aplicar_canny(str(DADOS / 'videos' / 'video_cinza.mp4'), 'video_canny.mp4')
