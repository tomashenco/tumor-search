import numpy as np
from random import choice

from dataset import Dataset
from cnn import target_size

from settings import image_size, num_classes, epoch_size, batch_size


class TrainDataset(Dataset):
    def __init__(self, path):
        Dataset.__init__(self, path)

    def iterate_data(self):
        """
        Iterates minibatches for training of image + mask

        :yield: tuple of input and target matrices
        """
        inputs = np.empty((batch_size, 1, image_size, image_size), dtype=np.float32)
        targets = np.empty((batch_size, num_classes, 2, target_size(image_size), target_size(image_size)),
                           dtype=np.float32)

        for i in xrange(epoch_size):
            # Initialise outputs
            inputs.fill(0.0)
            targets.fill(0.0)

            for index in xrange(batch_size):
                patient = choice(self.patients)
                image, mask = patient.get_random_data()
                inputs[index] = image
                targets[index, 0, 0] = mask
                targets[index, 0, 1] = 1.0 - mask

            yield inputs, targets
