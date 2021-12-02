import csv
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
    

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    # CODE GOES HERE
    
    return n, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():

    n, features, labels = readData('data/heart.csv')

    # CODE GOES HERE

    return (trainingData, testingData)


###################################################


trainingData, testingData = prepData()

net = network.Network([9,10,2])
net.SGD(trainingData, 10, 10, .1, test_data = testingData)


       