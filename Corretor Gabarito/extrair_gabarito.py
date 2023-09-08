import cv2
import numpy as np


def extrair_maior_ctn(img):
    """
    Realizando o processamento da imagem para extrair uma região de interesse ROI do gabarito
    """

    # Convertendo a imagem de cores para tons de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Operação de limiarização adaptativa para binarizar a imagem
    img_th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 12)

    # Criando uma mascara para realizar a dilatação da imagem e encontrando os contornos
    kernel = np.ones((2, 2), np.uint8)
    img_dil = cv2.dilate(img_th, kernel)
    coutours, hi = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Identificando o maior contorno com base na área e obtendo as coordenadas do retângulo delimitador
    maior_ctn = max(coutours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(maior_ctn)
    bbox = [x, y, w, h]

    # recortando e redimensionando a região de interesse ROI para o tamanho 400x500
    recorte = img[y:y+h, x:x+w]
    recorte = cv2.resize(recorte, (400, 500))

    return recorte, bbox
