import os

from patient import Patient
from settings import main_structures_file, main_structure_name, path_to_training_set, patient_structure_file, \
    patient_contours, patient_pngs


def my_key(s):
    return int(s.split('.')[0])


class TrainDataset(object):
    def __init__(self):
        self.patients = [patient for patient in self.prepare_patients()]

    @staticmethod
    def get_aliases():
        with open(main_structures_file, 'r') as f:
            whole_file = f.readlines()

        aliases = [line for line in whole_file if main_structure_name in line][0].strip().split('|')
        return aliases

    @staticmethod
    def read_test_structure(file_name, aliases):
        with open(file_name, 'r') as f:
            whole_file = f.readlines()
        processed = whole_file[0].strip().split('|')

        correct_index = 0
        for i, test_name in enumerate(processed):
            for name in aliases:
                if test_name == name:
                    correct_index = i + 1
                    break

        return correct_index

    def prepare_patients(self):
        alias = self.get_aliases()
        patients_list = [folder for folder in os.listdir(path_to_training_set)
                         if os.path.isdir(os.path.join(path_to_training_set, folder))]

        for patient in patients_list:
            index = self.read_test_structure(os.path.join(path_to_training_set, patient, patient_structure_file), alias)

            patient_path = os.path.join(path_to_training_set, patient)
            image_paths_unsorted = [path for path in os.listdir(os.path.join(patient_path, patient_pngs))]
            image_paths = sorted(image_paths_unsorted, key=my_key)

            contours_paths_unsorted = [path for path in os.listdir(os.path.join(patient_path, patient_contours))
                                       if int(path.split('.')[1]) == index]
            contours_paths = sorted(contours_paths_unsorted, key=my_key)

            for i, element in enumerate(image_paths):
                try:
                    if not contours_paths[i].split('.')[0] == element.split('.')[0]:
                        contours_paths.insert(i, None)
                except IndexError:
                    contours_paths.insert(i, None)

            assert(len(contours_paths) == len(image_paths))

            yield Patient(patient_path, contours_paths, image_paths)


train_dataset = TrainDataset()
# for patient in train_dataset.patients:
#     print patient.main_path
#     print patient.contours
#     print patient.images
