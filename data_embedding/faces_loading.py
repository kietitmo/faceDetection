import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class ImagesLoader:
    def __init__(self, dir):
        self.directory = dir
        self.X = []
        self.Y = []

    def load_faces(self, sub_dir):
        image_data = []
        path = os.path.join(self.directory, sub_dir)
        if os.path.isdir(path):
            for file_name in os.listdir(path):
                image_path = os.path.join(path, file_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image_data.append(image)
        return image_data

    def load_classes(self):
        for folder_name in os.listdir(self.directory):
            faces = self.load_faces(folder_name)
            labels = [folder_name for _ in range(len(faces))]
            print(f"Loaded successfully: {len(labels)} image(s) of {folder_name}")
            self.X.extend(faces)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        plt.figure(figsize=(18, 16))
        for num, image in enumerate(self.X):
            number_cols = 20
            number_rows = len(self.Y) // number_cols + 1
            plt.subplot(number_rows, number_cols, num + 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.axis('off')