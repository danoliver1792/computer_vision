import cv2
import mediapipe as mp
try:
    video = cv2.VideoCapture(0)

    # mapping the hands
    hand = mp.solutions.hands

    # maximum number of hands for the algorithm to recognize, drawing the lines of the hands
    Hand = hand.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    while True:
        check, img = video.read()
        if not check or img is None:
            continue

        # converting the image from BGR to RGB for processing in MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)
        hands_points = results.multi_hand_landmarks

        # returning the coordinates for each point
        if hands_points:
            for points in hands_points:
                print(points)

        cv2.imshow('Imagem', img)
        cv2.waitKey(1)
except Exception as e:
    print(f'Error: {e}')
