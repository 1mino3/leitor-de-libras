import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import threading

cap = None
hands = None
model = None
data = None
classes = ['A', 'B', 'C', 'D', 'E']

def processar_frames():
    global panel, cap, hands, model, data, classes

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        handsPoints = results.multi_hand_landmarks
        h, w, _ = img.shape
        if handsPoints is not None:
            for hand in handsPoints:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

                try:
                    imgCrop = img[y_min-50:y_max+50,x_min-50:x_max+50]
                    imgCrop = cv2.resize(imgCrop,(224,224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    data[0] = normalized_image_array
                    prediction = model.predict(data)
                    indexVal = np.argmax(prediction)
                    cv2.putText(img,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)

                except:
                    continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        panel.img = img
        panel.configure(image=img)
        panel.image = img

def iniciar_leitor():
    global cap, hands, model, data

    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(max_num_hands=1)
    model = load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    threading.Thread(target=processar_frames, daemon=True).start()

app = tk.Tk()
app.title("Leitor de Libras")
app.geometry("800x600")

title_label = tk.Label(app, text="Leitor de Libras", font=("Arial", 20))
title_label.pack(side=tk.TOP, pady=20)

start_button = tk.Button(app, text="Iniciar", command=iniciar_leitor)
start_button.pack()

panel = tk.Label(app)
panel.pack()

app.mainloop()