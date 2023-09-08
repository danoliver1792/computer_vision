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
        h, w, _ = img.shape
        array_points = []

        # returning the coordinates for each point
        if hands_points:
            for points in hands_points:

                # drawing the points
                mp_draw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)

                # list to store the coordinates of the points
                landmarks = []
                for landmark in points.landmark:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((cx, cy))

                # getting the points of the fingers, except the thumb
                fingers = [8, 12, 16, 20]
                cont = 0

                # checking if the finger is bent
                if landmarks:

                    # for the thumb
                    if landmarks[4][0] < landmarks[2][0]:
                        cont += 1
                    for x in fingers:
                        if landmarks[x][1] < landmarks[x - 2][1]:
                            cont += 1

                # showing the information
                cv2.putText(img, str(cont), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5)

        cv2.imshow('Imagem', img)
        cv2.waitKey(1)

except Exception as e:
    print(f'Error: {e}')
