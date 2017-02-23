from dataset import TrainDataset

train_dataset = TrainDataset()
for patient in train_dataset.patients:
    patient.iterate_data(train_dataset.get_max_size())