import lasagne


def test(network, test_dataset):
    """
    Tests the network by predicting tumors for test dataset and comparing them to ground truth
    It will print out the results

    :param network: lasagne model
    :param test_dataset: TestDataset object
    """


def compare_prediction(prediction, truth):
    """
    Compares one prediction to associated ground truth and returns F1 score, precision and recall
    False positive will be the area that should not have been found
    False negative will be the area that should have been found but was not

    :param prediction: image
    :param truth: image
    :return: tuple of F1, precision and recall
    """

