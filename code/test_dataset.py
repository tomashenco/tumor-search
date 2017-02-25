from dataset import Dataset


class TestDataset(Dataset):
    def __init__(self, path):
        Dataset.__init__(self, path)

    def iterate_data(self):
        """
        Iterates pairs for testing of image + mask

        :yield: tuple
        """
        for patient in self.patients:
            for image, mask in patient.get_all_data():
                yield image, mask
