import numpy as np

from dataset import Dataset

from settings import image_size


class TestDataset(Dataset):
    def __init__(self, path):
        Dataset.__init__(self, path)

    def iterate_data(self):
        """
        Iterates pairs for testing of image + mask

        :yield: tuple
        """
        input_mat = np.empty((1, 1, image_size, image_size), dtype=np.float32)
        for patient in self.patients:
            for image, mask in patient.get_all_data():
                input_mat[0] = image
                yield input_mat, mask
