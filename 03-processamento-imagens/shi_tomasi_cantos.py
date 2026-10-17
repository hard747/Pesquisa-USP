import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DADOS = Path(__file__).resolve().parent.parent / "dados"

# Detector de cantos Shi-Tomasi (Good Features to Track)
img = cv2.imread(str(DADOS / 'exemplos' / 'imagem_cena.jpg'), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
for x, y in np.int32(corners).reshape(-1, 2):
    cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
