
# coding: utf-8

# In[11]:

#################################
# Your name:Yonatan Gertskin
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib.pyplot as plt
import  matplotlib
"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper():
    mnist = fetch_mldata('MNIST original')
    print("TEAT")
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

   

def SGD(data ,labels ,C , eta_0 ,T):
    """
    Implements Hinge loss using SGD.
    returns: nd array of shape (data.shape[1],) or (data.shape[1],1) representing the final classifier
    """
    # TODO: Implement me
    m, n = data.shape

    ixs = np.arange(m)
    w = np.zeros(n)
    for i, t in enumerate(np.arange(T)):
        i_t = np.random.choice(m, 1)[0]
    x_t = data[i_t]
    y_t = labels[i_t]
    p = y_t * np.dot(w, x_t)
    if p > 1:
        w = (1 - eta_0) * w
    else:
        w = (1 - eta_0) * w + C * eta_0 * y_t * x_t
    return w


#################################

# Place for additional code

def calc_accuracy(vector):
    misses = 0
    for i in range(len(validation_data)):
        if np.sign(np.dot(vector, norm_validation_data[i])) != validation_labels[i]:
            misses += 1
    return 1 - (misses / len(validation_data))

#get training data and labels    
train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

norm_train_data = np.array([x / np.linalg.norm(x) for x in train_data])
norm_test_data = np.array([x / np.linalg.norm(x) for x in test_data])
norm_validation_data = np.array([x / np.linalg.norm(x) for x in validation_data])


#Q2 Section a
print('Q2.a')
y1 = []
x1 = list([10.0 ** i for i in range(-5, 6, 1)])
for eta_0 in x1:
    accuracy1 = []
    for i in range(10):
        classifier = SGD(norm_train_data, train_labels,1 ,eta_0, 1000)
        accuracy1.append(calc_accuracy(classifier))
    acc = np.mean(accuracy1)
    print("\neta_0 = "+str(eta_0)+"\t\tacc = "+str(acc))
    y1.append(acc)
best_eta=float(x1[np.argmax(y1)])
plt.figure()
plt.semilogx(x1, y1)
#plt.axis([best_eta/100,best_eta*100,0,1])
plt.savefig('Q2_a')

print('\nbest_eta = '+str(best_eta))
#Q2 Section b
print('Q2.b')
y2 = []
x2 = list([10.0 ** i for i in range(-5, 6, 1)])
for c in x2:
    accuracy2 = []
    for i in range(10):
        classifier = SGD(norm_train_data, train_labels,c,best_eta, 1000)
        accuracy2.append(calc_accuracy(classifier))
    acc = np.mean(accuracy2)
    print("\nc = "+str(c)+"\t\tacc = "+str(acc))
    y2.append(acc)
best_c=float(x2[np.argmax(y2)])
plt.figure()
plt.semilogx(x2, y2)
#plt.axis([best_c/100,best_c*100,0,1])
plt.savefig('Q2_b')
print('\nbest_c = '+str(best_c))

#Q2 Section c
print('Q2.c')
full_set = SGD(norm_train_data, train_labels,best_c,best_eta, 20000)
plt.figure()
plt.imshow(np.reshape(full_set, (28, 28)), interpolation='nearest')
plt.savefig('Q2_c')
#Q2 Section d
print('Q2.d')
best_acc=calc_accuracy(classifier)
print("the acurracy of the best classifier is:"+str(np.round(best_acc,3)))

#################################


# In[ ]:



