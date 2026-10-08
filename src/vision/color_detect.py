"""Deteccao de blobs coloridos via segmentacao HSV (OpenCV)."""
import cv2
import numpy as np

# faixas HSV (OpenCV: H 0-179, S/V 0-255). vermelho "da a volta" no matiz,
# por isso tem duas faixas.
#  saturacao minima alta (150) para nao confundir com partes do robo que tem
# tons proximos (ex: o anel azul-claro do punho da RG2, H~101 S~101 -- bem
# menos saturado que o cubo azul puro, H~120 S~216).
COLOR_RANGES = {
    "red": [
        ((0, 150, 60), (10, 255, 255)),
        ((170, 150, 60), (180, 255, 255)),
    ],
    "blue": [
        ((110, 150, 40), (130, 255, 255)),
    ],
}

_MIN_AREA_PX = 15


_ERODE_KERNEL = np.ones((3, 3), np.uint8)


def _mask_for(hsv_image, color):
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv_image, np.array(lo), np.array(hi))
    # remove pixels de borda/anti-aliasing, que mudam de forma entre as duas
    # cameras e enviesam o centroide (critico para a triangulacao estereo)
    return cv2.erode(mask, _ERODE_KERNEL, iterations=1)


def detect_color_centroid(rgb_image, color):
    """Retorna (cx, cy) em pixels do maior blob da cor, ou None se nao achar."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask = _mask_for(hsv, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < _MIN_AREA_PX:
        return None
    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])
