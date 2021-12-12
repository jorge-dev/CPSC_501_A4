import csv
import sys
from datetime import datetime as dt
from pprint import pprint
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

# given a data point, mean, and standard deviation, returns the z-score


def standardize(x, mu, sigma):
    return ((x - mu)/sigma)

# rescales a value to a new between 0 and 1


def rescale(x, min, max):
    return (x - min)/(max - min)

##############################################


''' Encode and standardize the data gathered from the heart.csv file
    returns a tuple (trainingData, testingData),
    each of which is a zipped array of features and labels
 '''


def getDataFromSample(sample, dataStat):
    sbp = dataStat['sbp']
    tobacco = dataStat['tobacco']
    ldl = dataStat['ldl']
    adiposity = dataStat['adiposity']
    typea = dataStat['typea']
    obesity = dataStat['obesity']
    alcohol = dataStat['alcohol']
    age = dataStat['age']

    sbp_std = cv([standardize(float(sample[1]), sbp['mean'], sbp['std'])])
    tobacco_std = cv([standardize(
        float(sample[2]), tobacco['mean'], tobacco['std'])])
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


'''
    Calculates the mean and standard deviation of each feature in the data
'''


def calculateDataStats(data: dict):
    sbp = []
    tobacco = []
    ldl = []
    adiposity = []
    famhist = []
    typea = []
    obesity = []
    alcohol = []
    age = []

    for row in data:
        sbp.append(float(row.get('sbp')))
        tobacco.append(float(row.get('tobacco')))
        ldl.append(float(row.get('ldl')))
        adiposity.append(float(row.get('adiposity')))
        famhist.append(row.get('famhist'))
        typea.append(float(row.get('typea')))
        obesity.append(float(row.get('obesity')))
        alcohol.append(float(row.get('alcohol')))
        age.append(float(row.get('age')))

    sbp_stats = {
        'mean': np.mean(sbp),
        'std': np.std(sbp),
        'min': np.min(sbp),
        'max': np.max(sbp)
    }

    tobacco_stats = {
        'mean': np.mean(tobacco),
        'std': np.std(tobacco),
        'min': np.min(tobacco),
        'max': np.max(tobacco)
    }

    ldl_stats = {
        'mean': np.mean(ldl),
        'std': np.std(ldl),
        'min': np.min(ldl),
        'max': np.max(ldl)
    }

    adiposity_stats = {
        'mean': np.mean(adiposity),
        'std': np.std(adiposity),
        'min': np.min(adiposity),
        'max': np.max(adiposity)
    }

    typea_stats = {
        'mean': np.mean(typea),
        'std': np.std(typea),
        'min': np.min(typea),
        'max': np.max(typea)
    }

    obesity_stats = {
        'mean': np.mean(obesity),
        'std': np.std(obesity),
        'min': np.min(obesity),
        'max': np.max(obesity)
    }

    alcohol_stats = {
        'mean': np.mean(alcohol),
        'std': np.std(alcohol),
        'min': np.min(alcohol),
        'max': np.max(alcohol)
    }

    age_stats = {
        'mean': np.mean(age),
        'std': np.std(age),
        'min': np.min(age),
        'max': np.max(age)
    }

    print(sbp_stats)

    return {
        'sbp': sbp_stats,
        'tobacco': tobacco_stats,
        'ldl': ldl_stats,
        'adiposity': adiposity_stats,
        'typea': typea_stats,
        'obesity': obesity_stats,
        'alcohol': alcohol_stats,
        'age': age_stats}


# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple


def readData(filename):
    with open(filename, 'r') as csvfile:
        data = csv.DictReader(csvfile)
        dataStat = calculateDataStats(data)

        # reset the file pointer to the beginning of the file
        csvfile.seek(0)
        reader = csv.reader(csvfile)
        next(reader, None)
        n = 0
        features = []
        labels = []
        for row in reader:
            n += 1
            featureVec, label = getDataFromSample(row, dataStat)
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


'''
    Initializes a neural network with the given parameters if no arguments are given,
    otherwise initializes a neural network with from the given file
'''


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


'''
    Prints a message box in a standardized format and with a custom message
'''


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
