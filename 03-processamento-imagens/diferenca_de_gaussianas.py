import cv2
import numpy as np
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

def difference_of_gaussians(image, sigma1, sigma2):
    image = image.astype(np.float32)
    gauss1 = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma1)
    gauss2 = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma2)
    dog = gauss1 - gauss2
    return dog

image = cv2.imread(str(DADOS / 'exemplos' / 'imagem_cena.jpg'), cv2.IMREAD_GRAYSCALE)
dog_result = difference_of_gaussians(image, 2, 2.8)
print(dog_result)
