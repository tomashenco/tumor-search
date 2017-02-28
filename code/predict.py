from optparse import OptionParser
import sys
from theano import tensor as t
import theano
import lasagne
import numpy as np
import cv2
import csv

from test_dataset import TestDataset
from test import prepare_prediction
from cnn import file_cnn

from settings import path_to_challenge_set, image_size

parser = OptionParser()
parser.add_option('-w', '--weights', action='store', dest='weights_src', help="Name of model file with initial weights")


def predict(network, dataset):
    my_csv = open('result.csv', 'wb')
    wr = csv.writer(my_csv, quoting=csv.QUOTE_ALL)

    x = t.tensor4('x')
    y = lasagne.layers.get_output(network, x, deterministic=True)
    f = theano.function([x], y, allow_input_downcast=True)

    input_mat = np.empty((1, 1, image_size, image_size), dtype=np.float32)

    for patient_num, patient in enumerate(dataset.patients):
        name = patient.get_name()
        tags = patient.get_tags()
        for i, (image, mask) in enumerate(patient.get_all_data()):
            input_mat[0] = image
            dx, dy = tags[i]['scale']
            x0, y0, z0 = tags[i]['centre']

            prediction = prepare_prediction(f(input_mat))
            for contour in convert_img_to_contour(prediction):
                contour_flat = contour.reshape((-1))
                contour_flat[::2] = contour_flat[::2] * dx + x0
                contour_flat[1::2] = contour_flat[1::2] * dy + y0
                contour_final = map(int, contour_flat)

                wr.writerow([name, str(i+1)] + contour_final)

        print 'Done %i / %i patients' % (patient_num+1, len(dataset.patients))


def convert_img_to_contour(image):
    image = np.uint8(image * 255.0)
    edges = cv2.Canny(image, 50, 100)
    img, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        yield contour


if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)

    print 'Building model'
    network_model = file_cnn(options.weights_src, True)
    print 'Loading dataset'
    challenge_dataset = TestDataset(path_to_challenge_set)

    predict(network_model, challenge_dataset)
