import cv2
import numpy as np
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def harris_corner_detection(image_path, measure=0.04, kappa=0.04, sigma_derivative=1, sigma_intensity=1, threshold=0.01):
    # Step 1: Smoothing the image
    img = cv2.imread(image_path, 0)
    img = np.float32(img)
    smoothed_img = cv2.GaussianBlur(img, (0, 0), sigma_derivative)

    # Step 2: Computing the gradient of the image
    Ix = cv2.Sobel(smoothed_img, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(smoothed_img, cv2.CV_64F, 0, 1, ksize=5)

    # Step 3: Computing autocorrelation matrix
    # (ponderado por uma janela gaussiana; sem ela det(M) = 0 em todo pixel)
    A = cv2.GaussianBlur(Ix**2, (0, 0), sigma_intensity)
    B = cv2.GaussianBlur(Iy**2, (0, 0), sigma_intensity)
    C = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma_intensity)

    # Step 4: Computing corner strength
    det_M = A * B - C**2
    trace_M = A + B
    R = det_M - kappa * trace_M**2

    # Step 5: Non-maximum suppression
    R[R < threshold * R.max()] = 0

    # Step 6: Selecting output corners
    corners = np.transpose(np.nonzero(R))

    return corners

# Ejemplo de uso
corners = harris_corner_detection(str(DADOS / 'exemplos' / 'imagem_cena.jpg'))
print(corners)

