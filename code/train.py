from random import choice
import sys
from optparse import OptionParser
import numpy as np
import theano
import theano.tensor as t
import lasagne
import time

from dataset import TrainDataset
import cnn
from settings import num_epochs, num_classes, image_size


class Logger(object):
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
parser.add_option('-m', '--max_error', action='store_true', dest='max_error', default=False,
                  help="Train only for class with max error")


def categorical_crossentropy_logdomain(log_predictions, targets):
    return -t.sum(targets * log_predictions, axis=2, keepdims=True)


def train(weight_src, learning_rate, max_error):
    # Loading train dataset
    print 'Loading datasets'
    train_dataset = TrainDataset()

    # Prepare Theano variables for inputs and targets
    input_var = t.tensor4('inputs')
    target_var = t.tensor4('targets')

    # Build CNN model
    print 'Building model and compiling functions'
    batch_size = train_dataset.get_max_size()
    if weight_src is None:
        network = cnn.empty_cnn(num_classes, False, batch_size, image_size, input_var)
    else:
        network = cnn.file_cnn(weight_src, False, batch_size, image_size, input_var, num_classes)
    prediction = lasagne.layers.get_output(network)

    # Create a loss expression for training (categorical crossentropy)
    loss = categorical_crossentropy_logdomain(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training (Stochastic Gradient Descent with Nesterov momentum)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=float(learning_rate), momentum=0.9)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

    # Launch the training loop
    print 'Starting training'
    for epoch in range(1, num_epochs + 1):
        print '\n\n---- EPOCH %i ----\n' % epoch
        patient = choice(train_dataset.patients)
        train_err = 0
        train_batches = 0

        train_time = time.time()
        for inputs, targets in patient.iterate_data(batch_size):
            train_err += train_fn(inputs, targets)
            train_batches += 1

        print 'Training took %.3f s loss: %.5f' % (time.time() - train_time, train_err / train_batches)

        save('snapshot', lasagne.layers.get_all_param_values(network))
        save('snapshot_' + str(epoch).zfill(4), lasagne.layers.get_all_param_values(network))


def save(base_filename, params):
    model_filename = base_filename + '.npz'
    np.savez(model_filename, params)


if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    train(options.weights_src, options.learning_rate, options.max_error)
