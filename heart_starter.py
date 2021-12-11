import csv
import sys
from datetime import datetime as dt

import numpy as np

import network

'''
HELPFUL FACTS:	min		max		average		std.dev
sbp				101		218		138.3		20.5
tobacco			0		31.2	3.64		4.59
ldl				.98		15.33	4.74		2.07
adiposity		6.74	42.49	25.4		7.77
famhist			binary value (1 = Present, 0 = Absent)
typea			13		78		53.1		9.81
obesity			14.7	46.58	26.0		4.21
alcohol			0		147.19	17.0		24.5
age				15		64		42.8		14.6
'''

sbp = {
    'min': 101,
    'max': 218,
    'mean': 138.3,
    'std': 20.5
}

tabacco = {
    'min': 0,
    'max': 31.2,
    'mean': 3.64,
    'std': 4.59
}

ldl = {
    'min': .98,
    'max': 15.33,
    'mean': 4.74,
    'std': 2.07
}

adiposity = {
    'min': 6.74,
    'max': 42.49,
    'mean': 25.4,
    'std': 7.77
}

typea = {
    'min': 13,
    'max': 78,
    'mean': 53.1,
    'std': 9.81

}

obesity = {
    'min': 14.7,
    'max': 46.58,
    'mean': 26.0,
    'std': 4.21
}

alcohol = {
    'min': 0,
    'max': 147.19,
    'mean': 17.0,
    'std': 24.5
}

age = {
    'min': 15,
    'max': 64,
    'mean': 42.8,
    'std': 14.6
}


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

# given a data point, mean, and standard deviation, returns the z-score


def standardize(x, mu, sigma):
    return ((x - mu)/sigma)


def rescale(x, min, max):
    return (x - min)/(max - min)

##############################################


def getDataFromSample(sample):
    rowName = cv(int(sample[0]))
    sbp_std = cv([standardize(float(sample[1]), sbp['mean'], sbp['std'])])
    tobacco_std = cv([standardize(
        float(sample[2]), tabacco['mean'], tabacco['std'])])
    ldl_std = cv([standardize(float(sample[3]), ldl['mean'], ldl['std'])])
    adiposity_std = cv([standardize(
        float(sample[4]), adiposity['mean'], adiposity['std'])])

    if sample[5] == "Present":
        famhist = cv([1])
    else:
        famhist = cv([0])

    typea_std = cv(
        [standardize(float(sample[6]), typea['mean'], typea['std'])])
    obesity_std = cv([standardize(
        float(sample[7]), obesity['mean'], obesity['std'])])
    alcohol_std = cv([standardize(
        float(sample[8]), alcohol['mean'], alcohol['std'])])
    age_std = cv([rescale(float(sample[9]), age['min'], age['max'])])

    features = np.concatenate(
        (sbp_std, tobacco_std, ldl_std, adiposity_std, famhist, typea_std, obesity_std, alcohol_std, age_std), axis=0)

    label = int(sample[-1])

    return (features, label)


# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        n = 0
        features = []
        labels = []
        for row in reader:
            n += 1
            featureVec, label = getDataFromSample(row)
            features.append(featureVec)
            labels.append(label)

    # CODE GOES HERE
    print(f"Number of data points: {n}")

    return n, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    n, features, labels = readData('data/heart.csv')

    ntrain = int(n * 5/6)
    ntest = n - ntrain

    # print(features[0])
    # print(labels[0])

    # split into training and testing data
    trainingFeatures = features[:ntrain]
    # training labels should be in onehot form
    trainingLabels = [onehot(label, 2) for label in labels[:ntrain]]

    print(f"Number of training samples: {ntrain}")

    testingFeatures = features[ntrain:]
    testingLabels = labels[ntrain:]
    print(f"Number of testing samples: {ntest}")
    trainingData = zip(trainingFeatures, trainingLabels)
    testingData = zip(testingFeatures, testingLabels)
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


input_neurons = 9
hidden_neurons = [10]
output_neurons = 2
sizes = [input_neurons] + hidden_neurons + [output_neurons]

epochs = 10
mini_batch_size = 10
eta = .1

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

network.saveToFile(net, "neural-nets/part3.pkl")
