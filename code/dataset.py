import os
import numpy as np
from random import choice

from patient import Patient
from cnn import target_size
from settings import main_structures_file, main_structure_name, path_to_training_set, patient_structure_file, \
    patient_contours, patient_pngs, patient_auxiliary, ct_tags, image_size, num_classes, epoch_size, batch_size


def my_key(s):
    return int(s.split('.')[0])


class TrainDataset(object):
    def __init__(self):
        self.__patients = [single_patient for single_patient in self.prepare_patients()]

    @staticmethod
    def get_aliases():
        """
        Get all possible aliases used

        :return: list of strings
        """
        with open(main_structures_file, 'r') as f:
            whole_file = f.readlines()

        # Multiple lines in form <radiomics_gtv|radiomics_gtv|Radiomics_gtv...>
        aliases = [line.strip().split('|') for line in whole_file if main_structure_name in line][0]
        return aliases

    @staticmethod
    def read_test_structure(file_name, aliases):
        """
        Get appropriate index for contour files

        :param file_name: path to file
        :param aliases: list of strings
        :return: single integer
        """
        with open(file_name, 'r') as f:
            whole_file = f.readlines()

        # One line in form of <body|Esophagus|lung|radiomics_gtv>
        processed = whole_file[0].strip().split('|')

        correct_index = 0
        for i, test_name in enumerate(processed):
            for name in aliases:
                if test_name == name:
                    correct_index = i + 1
                    break

        return correct_index

    @staticmethod
    def read_auxiliary(file_name, tags):
        """
        Reads auxiliary data and returns relevant tag values

        :param file_name: path to the file
        :param tags: list of tags to be read
        :return: dictionary of tag values
        """
        with open(file_name, 'r') as f:
            whole_file = f.readlines()

        # Multiple lines in form <(0018.0050),2.5>
        relevant_tags = {}
        for line in whole_file:
            for tag_name, tag_value in tags.iteritems():
                if tag_value in line:
                    relevant_tags[tag_name] = [float(value) for value in line.strip().split(',')[1:]]

        return relevant_tags

    def prepare_patients(self):
        """
        Prepares data for all patients by reading related files and structuring them as a list of Patient objects

        :yield: single Patient object
        """
        alias = self.get_aliases()
        patients_list = [folder for folder in os.listdir(path_to_training_set)
                         if os.path.isdir(os.path.join(path_to_training_set, folder))]

        for single_patient in patients_list:
            index = self.read_test_structure(os.path.join(path_to_training_set, single_patient, patient_structure_file),
                                             alias)

            patient_path = os.path.join(path_to_training_set, single_patient)

            # Return image, contour paths and CT tags sorted and of the same length for easier loading
            image_paths = [path for path in sorted(os.listdir(os.path.join(patient_path, patient_pngs)), key=my_key)]

            contours_paths = [path for path in sorted(os.listdir(os.path.join(patient_path, patient_contours)),
                                                      key=my_key) if int(path.split('.')[1]) == index]

            # Fill in absent contour files with Nones
            for i, element in enumerate(image_paths):
                try:
                    if not contours_paths[i].split('.')[0] == element.split('.')[0]:
                        contours_paths.insert(i, None)
                except IndexError:
                    contours_paths.insert(i, None)

            tags = [self.read_auxiliary(os.path.join(patient_path, patient_auxiliary, path), ct_tags) for path in
                    sorted(os.listdir(os.path.join(patient_path, patient_auxiliary)), key=my_key)]

            assert(len(contours_paths) == len(image_paths))
            assert(len(tags) == len(image_paths))

            yield Patient(patient_path, contours_paths, image_paths, tags)

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
                patient = choice(self.__patients)
                image, mask = patient.get_random_data()
                inputs[index] = image
                targets[index, 0, 0] = mask
                targets[index, 0, 1] = 1.0 - mask

            yield inputs, targets
