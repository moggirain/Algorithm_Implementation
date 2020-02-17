#!/usr/bin/python3
# coding: utf-8

# In[ ]:


import numpy as np
#TODO: understand that you should not need any other imports other than those already in this file; if you import something that is not installed by default on the csug machines, your code will crash and you will lose points

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "./" #TODO: if you are working somewhere other than the csug server, change this to the directory where a7a.train, a7a.dev, and a7a.test are on your 


# In[ ]:


#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value # put the parsed feature as the index, and according value into the array 
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    train_weight = []
    max_accuracy = 0.0
    dev_accuracy = []
    for j in range(args.iterations):
        for i in range(len(train_ys)):
            if train_ys[i] * np.dot(weights,train_xs[i]) <= 0:
                weights = weights + args.lr * np.dot(train_ys[i],train_xs[i]) # update the weights if misclassification
        # store weights for each iteration 
        train_weight.append(weights)
    
    if not args.nodev: # using the devset 
            weights = 0 # initate a weight 
            for weight in train_weight: # calcule the dev accuracy 
                accuracy = test_accuracy(weight,dev_ys,dev_xs)
                dev_accuracy.append(accuracy)
                if accuracy > max_accuracy:
                    weights = weight                            
    return weights

def test_accuracy(weights, test_ys, test_xs):
    # accuracy= correct classification by weight / total classification 
    accuracy = 0.0
    for i in range(len(test_ys)):
        if np.dot(weights,test_xs[i])*test_ys[i] >0: 
            accuracy+=1
    return accuracy/(len(test_ys))
#TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)


# In[ ]:


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.') # add argument to parse *arg
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))


if __name__ == '__main__':
    main()

