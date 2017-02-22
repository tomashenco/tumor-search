import os
import cv2
import numpy as np

from settings import patient_contours, patient_pngs, image_size, classes


class Patient(object):
    def __init__(self, path, contours, images):
        self.__main_path = path
        self.__contours = contours
        self.__images = images

    def get_images_length(self):
        return len(self.__images)

    def iterate_data(self, max_size):
        # Initialise outputs
        inputs = np.empty((max_size, 1, image_size, image_size), dtype=np.float32)
        targets = np.empty((max_size, classes, image_size, image_size), dtype=np.float32)

        inputs.fill(0.0)
        targets.fill(0.0)

        for index, image_name in enumerate(self.__images):
            inputs[index] = self.prepare_image(os.path.join(self.__main_path, patient_pngs, image_name))

        for index, contour_name in enumerate(self.__contours):
            print '*'*15
            if contour_name is not None:
                inputs[index] = self.prepare_mask(os.path.join(self.__main_path, patient_contours, contour_name))


    @staticmethod
    def prepare_image(path):
        image = cv2.imread(path, 0).astype(float)
        image = cv2.resize(image, (image_size, image_size))
        image /= np.amax(image)

        return image

    @staticmethod
    def prepare_mask(path):
        with open(path, 'r') as f:
            tumor_contours = f.read()

        print tumor_contours



