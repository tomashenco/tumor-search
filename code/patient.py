import os
import cv2
import numpy as np

from settings import patient_contours, patient_pngs, image_size, classes


class Patient(object):
    def __init__(self, path, contours, images, tags):
        self.__main_path = path
        self.__contours = contours
        self.__images = images
        self.tags = tags

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

        for index, (contour_name, tag) in enumerate(zip(self.__contours, self.tags)):
            if contour_name is not None:
                mask = self.prepare_mask(os.path.join(self.__main_path, patient_contours, contour_name), tag)
            else:
                mask = np.zeros((512, 512), dtype=float)

            targets[index] = mask

        yield inputs, targets

    @staticmethod
    def prepare_image(path):
        image = cv2.imread(path, 0).astype(float)
        image = cv2.resize(image, (image_size, image_size))
        image /= np.amax(image)

        return image

    @staticmethod
    def prepare_mask(path, tag):
        with open(path, 'r') as f:
            tumor_contours = f.read().strip().split()

        mask = np.zeros((image_size, image_size), dtype=float)
        dx, dy = tag['scale']
        x0, y0, z0 = tag['centre']

        for contour in tumor_contours:
            contour_3d = np.array([float(value) for value in contour.split(',')])
            contour_2d = np.delete(contour_3d, np.arange(2, contour_3d.size, 3))
            contour_2d[::2] = (contour_2d[::2] - x0) / dx
            contour_2d[1::2] = (contour_2d[1::2] - y0) / dy

            contour = contour_2d.reshape((-1, 2)).astype(int)
            cv2.fillConvexPoly(mask, contour, 1.0)

        return mask
