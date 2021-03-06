import sys
from optparse import OptionParser
import numpy as np
import theano
import theano.tensor as t
import lasagne
import time
import os

from train_dataset import TrainDataset
from test_dataset import TestDataset
import cnn
from test import test

from settings import num_epochs, num_classes, image_size, epoch_size, batch_size, models_path, path_to_training_set, \
    path_to_testing_set


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logfile.txt', 'a')

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

sys.stdout = Logger()

parser = OptionParser()
parser.add_option('-w', '--weights', action='store', dest='weights_src', default=None,
                  help="Name of model file with initial weights")
parser.add_option('-r', '--learning_rate', action='store', dest='learning_rate', default=0.01,
                  help="Training learning rate")


def categorical_crossentropy_logdomain(log_predictions, targets):
    return -t.sum(targets * log_predictions, axis=2, keepdims=True)


def train(weight_src, learning_rate):
    # Loading train dataset
    print 'Loading datasets'
    train_dataset = TrainDataset(path_to_training_set)
    test_dataset = TestDataset(path_to_testing_set)

    # Prepare Theano variables for inputs and targets
    ftensor5 = t.TensorType('float32', (False,) * 5)
    input_var = t.tensor4('inputs')
    target_var = ftensor5('targets')

    # Build CNN model
    print 'Building model and compiling functions'
    if weight_src is None:
        network = cnn.empty_cnn(num_classes * 2, False, batch_size, image_size, input_var)
    else:
        network = cnn.file_cnn(weight_src, False, batch_size, image_size, input_var, num_classes * 2)
    prediction = lasagne.layers.get_output(network)

    # Create a loss expression for training (categorical crossentropy)
    loss = categorical_crossentropy_logdomain(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training (Stochastic Gradient Descent with Nesterov momentum)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=float(learning_rate), momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and
    # returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Launch the training loop
    print 'Starting training'
    for epoch in range(1, num_epochs + 1):
        print '\n\n---- EPOCH %i ----\n' % epoch
        train_err = 0

        train_time = time.time()
        for inputs, targets in train_dataset.iterate_data():
            train_err += train_fn(inputs, targets)

        print 'Training took %.3f s loss: %f' % (time.time() - train_time, train_err / epoch_size)

        save('snapshot', lasagne.layers.get_all_param_values(network))
        save('snapshot_' + str(epoch).zfill(4), lasagne.layers.get_all_param_values(network))

        # Test the model
        network_model = cnn.pretrained_cnn(lasagne.layers.get_all_param_values(network), True)
        print 'Calculating score'
        test(network_model, test_dataset)


def save(base_filename, params):
    model_filename = base_filename + '.npz'
    np.savez(os.path.join(models_path, model_filename), params)


if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    train(options.weights_src, options.learning_rate)
