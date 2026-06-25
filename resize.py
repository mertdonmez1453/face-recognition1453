import os
import cv2
import numpy as np
import pandas
import sklearn


def resize_dataset(input_dir, output_dir):
    for person_name in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        save_person_path = os.path.join(output_dir, person_name)
        os.makedirs(save_person_path, exist_ok=True)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print("Skipping unreadable file:", img_path)
                continue

            resized = cv2.resize(img, (224, 224)) #224, hyperparameter
            save_path = os.path.join(save_person_path, img_name)
            cv2.imwrite(save_path, resized)


if __name__ == "__main__":
    source_dir = "datasets/Faces-Datasets"
    target_dir = "datasets/Faces-Datasets-scaled"
    resize_dataset(source_dir, target_dir)


