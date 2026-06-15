import numpy as np

import cv2
import pandas
import os 


##is it allowed
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

##some photos are not clean, should i clean manually?
def resize(input_dir,output_dir):
    for person_name in os.listdir(input_dir):
        person_path = os.path.join(input_dir,person_name)
        save_person_path = os.path.join(output_dir, person_name)
        os.makedirs(save_person_path, exist_ok=True)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            print(img_path)

            img = cv2.imread(img_path)

            if img is None:
                continue
            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            if len(faces) == 0:
                print("face is not found")
                continue
            
            ##all images has only 1 face
            x, y, w, h = faces[0]
            
            margin = int(0.2 * w)

            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)

            face_crop = img[y1:y2, x1:x2]

            face_crop = cv2.resize(face_crop, (224,224))

            save_path = os.path.join(save_person_path, img_name)
            cv2.imwrite(save_path, face_crop)











