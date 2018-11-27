#################################
# Your name:Yonatan Gertskin
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_mldata
from scipy.io import loadmat
import sklearn.preprocessing

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def perceptron(data, labels):
    """
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the perceptron classifier
    """
    # TODO: Implement me
    w = np.zeros(len(data[0])) #initialize w1 as 0
    for i in range(data):
        if np.sign(np.dot(w, data[i] )) != labels[i]:
            w += np.dot(data[t], labels[t])    #for each new measurment update w[t]
    return w

#################################

# Place for additional code

def Calc_Accuracy(w_perceptron):
    misses = 0
    for i in range(len(test_data)):
        if np.sign(np.dot(w_perceptron, test_data[i])) != test_labels[i]:
            misses += 1
    return ( 1 - count_miss/len(test_data))

def Mistakes_Indices(w_perceptron):
    mistakes=[]
    for i in range(len(test_data)):
        if len(mistakes)==2:
            break
        if np.sign(np.dot(w_perceptron, test_data[i])) != test_labels[i]:
            mistakes.append(i)
    return mistakes    

def helper():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    neg, pos = 0,8
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

#get training data and labels    
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
normalized_train_data = np.array([x / np.linalg.norm(x) for x in train_data])
normalized_test_data = np.array([x / np.linalg.norm(x) for x in test_data])

#Q1 Section a   
mean_accuracy_per_n=[]
for n in np.array([5, 10, 50, 100, 500, 1000, 5000]):
    order_of_inputs =np.random.permutation(n)
    accuracy = []
    for i in range(100):
        w_perceptron = perceptron(normalized_train_data[order_of_inputs], train_labels[order_of_inputs])
        accuracy.append(Calc_Accuracy(w_perceptron))
        
    five_percentile= np.percentile(accuracy,np.array(5))
    nintyfive_percentile= np.percentile(accuracy,np.array(95))
    mean = np.mean(accuracy)
    print("n =\t"+ str(n) + "\t\tmean =\t" + str(round(mean,3)) +"\t\t5% =\t" +str(round(five,3)) + "\t\t\n95% =\t" + str(round(ninty_five,3)) + "\n")

#Q1 Section b
w_full_set = perceptron(normalized_train_data, train_labels)
plt.imshow(np.reshape(w_full_set, (28, 28)), interpolation='nearest')

#Q1 Section c
acc_full_training_set=Calc_Accuracy(w_full_set)
print(round(acc_full_training_set, 3))
    
#Q1 Section d
indices=Mistakes_Indices(w_full_set)
for i in indices:
    plt.imshow(np.reshape(test_data_unscaled[i], (28, 28)), interpolation='nearest')
    
#################################