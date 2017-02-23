from random import choice
from dataset import TrainDataset
from settings import num_epochs


train_dataset = TrainDataset()
for i in range(num_epochs):
    patient = choice(train_dataset.patients)
    for inputs, targets in patient.iterate_data(train_dataset.get_max_size()):
        pass
