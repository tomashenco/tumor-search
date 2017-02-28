from math import sqrt
import numpy as np
import theano.tensor as t
import lasagne

conv_0_filters = 16
conv_1_filters = 16
conv_2_filters = 32
conv_3_filters = 64
conv_4_filters = 128

network_scale = 8
total_padding = 32
receptive_field_size = 72


def softmax4d(x):
    e_x = t.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def softmax(x):
    e_x = t.exp(x - x.max(axis=2, keepdims=True))
    return e_x / e_x.sum(axis=2, keepdims=True)


def log_softmax(x):
    xdev = x - x.max(axis=2, keepdims=True)
    return xdev - t.log(t.sum(t.exp(xdev), axis=2, keepdims=True))


def empty_cnn(num_classes, test_mode, batch_size=None, image_size=None, input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(batch_size, 1, image_size, image_size), input_var=input_var)

    # Convolutional layers #0
    if test_mode:
        network = lasagne.layers.pad(network, 2, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_0_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    if test_mode:
        network = lasagne.layers.pad(network, 2, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_0_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Convolutional layers #1
    if test_mode:
        network = lasagne.layers.pad(network, 1, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_1_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    if test_mode:
        network = lasagne.layers.pad(network, 1, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_1_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Convolutional layers #2
    if test_mode:
        network = lasagne.layers.pad(network, 1, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_2_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    if test_mode:
        network = lasagne.layers.pad(network, 1, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_2_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    # Max-pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Convolutional layers #3
    if test_mode:
        network = lasagne.layers.pad(network, 1, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_3_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    if test_mode:
        network = lasagne.layers.pad(network, 1, 0.0)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_3_filters, filter_size=(3, 3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    # Convolutional layer #4
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=conv_4_filters, filter_size=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain=sqrt(2)))

    # Final convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=num_classes, filter_size=(1, 1), nonlinearity=lasagne.nonlinearities.identity)

    network = lasagne.layers.ReshapeLayer(network, ([0], num_classes / 2, 2, [2], [3]))

    if test_mode:
        network = lasagne.layers.NonlinearityLayer(network, softmax)
    else:
        network = lasagne.layers.NonlinearityLayer(network, log_softmax)

    return network


def pretrained_cnn(model_weights, test_mode, batch_size=None, image_size=None, input_var=None, num_classes=None):
    model_classes = model_weights[len(model_weights) - 1].shape[0]
    if num_classes is None:
        num_classes = model_classes

    network = empty_cnn(num_classes, test_mode, batch_size, image_size, input_var)

    weights = lasagne.layers.get_all_param_values(network)
    if model_classes == num_classes:
        weights = model_weights
    else:
        print "Warning: different number of output classes, intializing model without last layer"
        weights[:-2] = model_weights[:-2]

    lasagne.layers.set_all_param_values(network, weights)

    return network


def file_cnn(model_filename, test_mode, batch_size=None, image_size=None, input_var=None, num_classes=None):
    model_weights = np.load(model_filename)
    model_weights = model_weights[model_weights.keys()[0]]

    network = pretrained_cnn(model_weights, test_mode, batch_size, image_size, input_var, num_classes)
    return network


def target_size(input_size):
    return (input_size - 2 * total_padding) / network_scale
