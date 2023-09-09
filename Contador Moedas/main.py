import cv2
import numpy as np
import tensorflow
from keras.models import load_model

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = load_model('Keras_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
classes = ['1 real', '25 cent', '50 cent']


def pre_process(img):
    """
    Pre-processing the image
    """
    # increasing the pixels of the image making it blurry and filtering beautiful edges
    img_pre = cv2.GaussianBlur(img, (5, 5), 3)
    img_pre = cv2.Canny(img_pre, 90, 140)
    kernel = np.ones((4, 4), np.uint8)

    # applying dilation and erosion
    img_pre = cv2.dilate(img_pre, kernel, iterations=2)
    img_pre = cv2.erode(img_pre, kernel, iterations=1)
    return img_pre


def detect_coin(img):
    """
    Currency recognition
    """
    # normalizing the image
    img_coin = cv2.resize(img, (224, 224))
    img_coin = np.asarray(img_coin)
    img_coin_norm = (img_coin.astype(np.float32) / 127.0) - 1
    data[0] = img_coin_norm
    prediction = model.predict(data)
    index = np.argmax(prediction)
    percent = prediction[0][index]
    classe = classes[index]
    return classe, percent


while True:
    _, img = video.read()
    img = cv2.resize(img, (640, 480))
    img_pre = pre_process(img)
    countors, h1 = cv2.findContours(img_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # looping through the contour variable to identify each individual object
    for cnt in countors:
        # validating the area of the object
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # training the neural network to recognize coins
            cut = img[y:y + h, x:x + w]
            classe, conf = detect_coin(cut)
            cv2.putText(img, str(classe), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    cv2.imshow('Image Pre', img_pre)
    cv2.waitKey(1)
