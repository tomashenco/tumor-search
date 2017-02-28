import lasagne
import theano
from theano import tensor as t
import numpy as np
from optparse import OptionParser
import sys
import cv2
import time

from cnn import file_cnn
from test_dataset import TestDataset

from settings import path_to_testing_set, image_size, threshold

parser = OptionParser()
parser.add_option('-w', '--weights', action='store', dest='weights_src', help="Name of model file with initial weights")


def test(network, test_dataset):
    """
    Tests the network by predicting tumors for test dataset and comparing them to ground truth
    It will print out the results

    :param network: lasagne model
    :param test_dataset: TestDataset object
    """
    x = t.tensor4('x')
    y = lasagne.layers.get_output(network, x, deterministic=True)
    f = theano.function([x], y, allow_input_downcast=True)

    precision_list = []
    recall_list = []
    weights_list = []

    test_time = time.time()

    for image, mask in test_dataset.iterate_data():
        precision, recall, weight = compare_prediction(prepare_prediction(f(image)), mask)
        precision_list.append(precision)
        recall_list.append(recall)
        weights_list.append(weight)

    total_precision = np.average(precision_list, weights=weights_list) * 100.0
    total_recall = np.average(recall_list, weights=weights_list) * 100.0

    if total_precision != 0 and total_recall != 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0.0

    print 'Testing took %.3f s, F1: %.3f %%, Precision: %.3f %%, Recall: %.3f %%' % (time.time() - test_time, total_f1,
                                                                                     total_precision, total_recall)


def prepare_prediction(prediction):
    """
    Resizes prediction matrix and applies threshold

    :param prediction: network output
    :return: image
    """
    prediction_small = prediction[0, 0, 0]
    prediction = cv2.resize(prediction_small, (image_size, image_size), interpolation=cv2.INTER_AREA)
    prediction[prediction >= threshold] = 1.0
    prediction[prediction < threshold] = 0.0

    return prediction


def compare_prediction(prediction, truth):
    """
    Compares one prediction to associated ground truth and returns precision, recall and weight of the example
    Weight is how the example is relevant. That will be measured by percentage of the area of the image that is the
    tumor or was predicted to be a tumor (whichever is higher)

    :param prediction: image
    :param truth: image
    :return: tuple of precision, recall and weight
    """
    weight = max(cv2.countNonZero(prediction), cv2.countNonZero(truth))

    tp_area = np.zeros((image_size, image_size))
    tp_area[np.bitwise_and(prediction == 1.0, truth == 1.0)] = 1.0
    tp = float(cv2.countNonZero(tp_area))

    fp_area = np.zeros((image_size, image_size))
    fp_area[np.bitwise_and(prediction == 1.0, truth == 0.0)] = 1.0
    fp = float(cv2.countNonZero(fp_area))

    fn_area = np.zeros((image_size, image_size))
    fn_area[np.bitwise_and(prediction == 0.0, truth == 1.0)] = 1.0
    fn = float(cv2.countNonZero(fn_area))

    if tp == 0 and fp == 0:
        precision = 1.0
    else:
        precision = tp / (tp + fp)
    if tp == 0 and fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    return precision, recall, weight


if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)

    print 'Building model'
    network_model = file_cnn(options.weights_src, True)
    print 'Loading dataset'
    testing_dataset = TestDataset(path_to_testing_set)
    print 'Calculating score'
    test(network_model, testing_dataset)
