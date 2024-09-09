import cv2
import mediapipe as mp
import time
import tkinter as tk
from tkinter import ttk

# Configurações do Mediapipe
video = cv2.VideoCapture(0)
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape
    pontos = [] # pontos: pontos[numero][eixo]
    if handsPoints:
        for points in handsPoints:
             # mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS) # desenho das conexões entre os pontos (traços)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                # cv2.putText(img, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # desenho dos pontos
                cz = cord.z
                pontos.append((cx, cy, cz))

        letra = "" # Variável onde a letra detectada será armazenada
        if ( # Letra A
            pontos[8][1] > pontos[5][1] and
            pontos[12][1] > pontos[9][1] and
            pontos[16][1] > pontos[13][1] and
            pontos[20][1] > pontos[17][1] and
            pontos[4][1] < pontos[5][1]
            ):
            print("Letra: A")
            letra = "A"


        elif ( # Letra B
            pontos[8][1] < pontos[7][1] and
            pontos[12][1] < pontos[11][1] and
            pontos[16][1] < pontos[15][1] and
            pontos[20][1] < pontos[19][1] and
            pontos[4][0] < pontos[1][0]
            ):
            print("Letra: B")
            letra = "B"


        elif ( # Letra C
            pontos[8][2] > pontos[12][2] and
            pontos[12][2] > pontos[16][2] and
            pontos[16][2] > pontos[20][2] and
            pontos[20][2] < pontos[16][2] and
            pontos[20][2] < pontos[12][2] and
            pontos[20][2] < pontos[18][2] and
            pontos[4][0] > pontos[3][0] and
            pontos[4][2] > pontos[20][2]
        ):
            print("Letra: C")
            letra = "C"


        elif ( # Letra D
            pontos[8][1] < pontos[10][1] and
            pontos[12][1] > pontos[10][1] and
            pontos[16][1] > pontos[14][1] and
            pontos[20][1] > pontos[18][1] and
            pontos[4][1] > pontos[15][1]

        ):
            print("Letra: D")
            letra = "D"


        elif ( # Letra E
            pontos[8][1] < pontos[5][1] and
            pontos[20][1] > pontos[19][1] and
            pontos[16][1] > pontos[15][1] and
            pontos[12][1] > pontos[11][1] and
            pontos[8][1] > pontos[7][1] and
            pontos[8][2] < pontos[9][2] and
            pontos[16][2] < pontos[13][2] and
            pontos[20][2] < pontos[17][2] and
            pontos[4][1] > pontos[5][1] and
            pontos[4][2] < pontos[2][2]

        ):
            print("Letra: E")
            letra = "E"

    cv2.imshow("Leitor de Libras", img)
    cv2.waitKey(1)
    if cv2.getWindowProperty("Leitor de Libras", cv2.WND_PROP_VISIBLE) < 1:
        break