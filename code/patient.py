import os
import cv2
import numpy as np
from random import randint

from cnn import target_size
from settings import patient_contours, patient_pngs, image_size


class Patient:
    def __init__(self, path, contours, images, tags):
        self.__main_path = path
        self.__contours = contours
        self.__images = images
        self.__tags = tags

    def get_name(self):
        return self.__main_path.split('/')[-1]

    def get_tags(self):
        return self.__tags

    def get_random_data(self):
        """
        Prepare a pair of image + mask

        :return: tuple
        """
        index = randint(0, len(self.__images) - 1)

        image_name = self.__images[index]
        image = self.prepare_image(os.path.join(self.__main_path, patient_pngs, image_name))

        contour_name = self.__contours[index]
        tag = self.__tags[index]
        if contour_name is not None:
            mask = self.prepare_mask(os.path.join(self.__main_path, patient_contours, contour_name), tag)
        else:
            mask = np.zeros((target_size(image_size), target_size(image_size)), dtype=float)

        mask = cv2.resize(mask, (target_size(image_size), target_size(image_size)), interpolation=cv2.INTER_AREA)

        return image, mask

    def get_all_data(self):
        """
        Prepare all pairs of image + mask

        :yield: tuple
        """

        for image_name, contour_name, tag in zip(self.__images, self.__contours, self.__tags):
            image = self.prepare_image(os.path.join(self.__main_path, patient_pngs, image_name))
            if contour_name is not None:
                mask = self.prepare_mask(os.path.join(self.__main_path, patient_contours, contour_name), tag)
            else:
                mask = np.zeros((image_size, image_size), dtype=float)

            yield image, mask

    @staticmethod
    def prepare_image(path):
        image = cv2.imread(path, 0).astype(float)
        image = cv2.resize(image, (image_size, image_size))
        image /= np.amax(image)

        return image

    @staticmethod
    def prepare_mask(path, tag):
        """
        Converts contour to mask

        :param path: path to the contour file
        :param tag: appropriate tags for the CT scan
        :return:
        """
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
