import sys
from datetime import datetime as dt
from time import sleep

import numpy as np

import network


# converts a 1d python list into a (1,n) row vector


def rv(vec):
    return np.array([vec])

# converts a 1d python list into a (n,1) column vector


def cv(vec):
    return rv(vec).T

# creates a (size,1) array of zeros, whose ith entry is equal to 1


def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)


#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']

    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    trainFeatures = [np.reshape(img, (784, 1)) for img in train_features]
    trainFeatures = [np.array(f) / 255.0 for f in trainFeatures]
    trainLabels = [onehot(l, 10) for l in train_labels]

    # need to rescale, flatten, convert testing labels to one-hot, and zip appropriate components together
    testFeatures = [np.reshape(img, (784, 1)) for img in test_features]
    testFeatures = [np.array(f) / 255.0 for f in testFeatures]
    testLabels = test_labels

    # zip the training data together
    trainingData = zip(trainFeatures, trainLabels)

    # zip the testing data together
    testingData = zip(testFeatures, testLabels)

    return (trainingData, testingData)


def loadNet(sizes, bias=None, weight=None):
    if (len(sys.argv) > 1):
        if sys.argv[1] == "--load":
            net = network.loadFromFile(sys.argv[2])
            print("Loaded net from file: ", sys.argv[2])
        else:
            print("Initializing new net")
            net = network.Network(sizes, bias, weight)

    else:
        print("Initializing new net")
        net = network.Network(sizes, bias, weight)

    return net


def print_msg_box(msg, indent=1, width=None, title=None):
    #  code from https://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)


###################################################################


input_neurons = 784
hidden_neurons = [25]
output_neurons = 10
sizes = [input_neurons] + hidden_neurons + [output_neurons]

epochs = 20
mini_batch_size = 10
eta = 2.5

message = f"""Epochs: {epochs} 
Mini-batch size: {mini_batch_size}
Step Size: {eta}
Number of Layers: {len(sizes)}
Number of Hidden Neurons: {len(sizes) - 2}
Number of Neurons in each Layer: {' ,'.join(str(x) for x in sizes)}"""

print_msg_box(message, 8, title="Hyperparameters")

trainingData, testingData = prepData()

net = loadNet(sizes)

start = dt.now()
net.SGD(trainingData, epochs, mini_batch_size, eta, test_data=testingData)
end = dt.now()
total_time = end - start
total_time_min, total_time_sec = divmod(total_time.total_seconds(), 60)

print(f"Total time taken: {total_time_min:.0f} min {total_time_sec:.2f} sec")

network.saveToFile(net, "neural-nets/part2.pkl")
