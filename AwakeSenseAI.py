import cvzone
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from ultralytics import YOLO

import uuid
import time
import os

# Capturing VideoData
cap = cv2.VideoCapture(0)

model = YOLO("../YOLO-Weights/yolov8l.pt")

image_path = os.path.join('data', 'images')
classNames = ["Awake", "Drowsy"]

# Increase this value to make the model to collect more pictures
ImagesCount = 20


while True:

    returnVal, img = cap.read()
    results = model(img, stream=True)

    # for i in results:
    #     boxes = i.boxes
    #     for box in boxes:
    #         x1, y1, x2, y2 = box.xyxy[0]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #
    #         # cls = int(box.cls[0])
    #         # currentClass = classNames[cls]
    #         # print(currentClass)
    #
    #         conf = math.ceil((box.conf[0]) * 100) / 100
    #         print(conf)
    #
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
    #         cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(y1, 30)), 1, 2, colorR=(0, 0, 0))

    # Creating loop for capturing custom data
    for label in classNames:
        print("Collecting images for {}".format(label))
        time.sleep(5)   # Increase this time to increase the interval time during the switch between Awake and Drowsy

        for imgNum in range(1, ImagesCount+1):
            print("{} Image #{}".format(label, imgNum))
            returnVal, img = cap.read()

            imgName = os.path.join(image_path, label+"."+str(uuid.uuid1())+".jpg")
            cv2.imwrite(imgName, img)
            cv2.imshow("CapturedImage", img)
            time.sleep(2)  # Increase this time to increase the interval time between capturing images

    cv2.imshow("AwakeSenseAI", img)
    cv2.waitKey(0)

