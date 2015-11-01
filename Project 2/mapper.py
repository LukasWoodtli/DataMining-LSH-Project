#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

## goal: use parallel stochastic gradient descent in a map-reduce environment
## mapper: in multiple mappers (in this project 10) we train separate models 
##         by applying support vector machines
## reducer: in a single reducer we avarage the model parameters of the mappers

## import requried packages
import sys
import numpy as np
import scipy as sc
import sklearn as sl
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
from sklearn import preprocessing

starttime = datetime.datetime.now()

## definition of transformation function
def transform(x_original):
    #x_original = np.append(x_original, 0)
    ## x_transformed = rbf_feature.fit_transform(x_original)
    ## x_transformed = additive.fit_transform(x_original)
    ## x_transformed = nystroem.fit_transform(x_original)
    ## x_transformed = preprocessing.scale(x_original)
    ## min_max_scaler = preprocessing.MinMaxScaler()
    ## x_transformed = min_max_scaler.fit_transform(x_original)
    scaler.fit(x_original)
    x_transformed = scaler.transform(x_original)
    return x_transformed

## Step 1: define values for parameters C, ETA and THETA
#DIMENSION = 1200        ## Dimension of the original data.
DIMENSION = 400        ## Dimension of the original data.
CLASSES = (-1, +1)      ## The classes that we are trying to predict.
ETA = 0.001             ## eta is the learning-rate parameter (small, positive real number)
                        ## the choice of eta affects the convergence
                        ##      -> if eta is too small, then convergence is slow
                        ##      -> if eta is too big, then the decision boundary will
                        ##         "dance around" and convergence will as well be slow
b = 1                   ##      -> if b is different from 0 then it can be included into the weight vector
                        ##         (at the cost of adding another dimension to the feature vector)
C = 2                   ## penalty for wrongly classifid points

#additive = AdditiveChi2Sampler()
#nystroem = Nystroem(n_components=DIMENSION)
#rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=DIMENSION)
scaler = StandardScaler()

if __name__ == "__main__":

    ## Step 2: Initial values for weight vector including
    ##          -> in this case wheight vector is initialized to zeros
    ##          -> in addition we make b part of the weight vector
    ##              -> b is the negative of a threshold on the dot product w.x
    w = np.array([0]*(DIMENSION))
    #w = np.array([0]*(DIMENSION+1))
    #w[len(w)-1] = b

    ## read from file
    #fp = open(r"C:\Users\mme\Desktop\training.txt", "r")
    fp = open(r"C:\Users\Michael\Desktop\training.txt", "r")
    #fp = open(r"C:\Dropbox\___DM_Project2\visual_test\visual_test_set_with_target.csv", "r")

    x_all = []
    label_all = []

    #for line in sys.stdin:
    for i, line in enumerate(fp):

        if i < 5000:
            
            ## Step 3: Take each line and split it up into
            ## label and set of features extracted from each image
            line = line.strip()
            (label, x_string) = line.split(" ", 1)
            #(label, x_string) = line.split(";", 1)
            label = int(label)
            x_original = np.fromstring(x_string, sep=' ')

            ## Step 4: Transformation of the features
            ## Expand the n-dimensional feature vector x with an additional
            ##         componant +1 according to the weight-vector
            ## => Qestion: Why +1?
            x = transform(x_original)  # Use our features.
            x_all.append(x)
            label_all.append(label)

            ## Step 5: If x is classified wrongly we compute the partial derivatives of f(w,b)
            ##         with respect to b and each component wj of the weight vector w.
            ##         Afterwards we need to adapt weight vector w. Since we want to
            ##         minimize f(w,b) we move b and the components wj in the
            ##         direction opposite to the direction of the gradient. The amount we
            ##         move each component is proportional to the derivative with respect
            ##         to that component.
            ##
            ##      -> if yi ( sum (wj.xij + b)) >= 1 then 0 else -yi.xij
            ##          -> if xi is classified correctly do nothing
            ##             otherwise weight vector w needs to be adapted
            #if (np.dot(label, np.dot(w, x)) >= 1):
            #print "before", w
            if not (label*(np.inner(w, x)) >= 1):
                
                ## (a) Compute partial derivatives
                delfdelwj = w + C * ((-1)*(label)*x)
                ## (b) Adapt weight vector w
                w = w - ETA * delfdelwj
                #w_strich = w + ETA * label * x
                #w = w_strich * min(1, 1 / (np.linalg.norm(w_strich) * math.sqrt(C)))
                
            else:
                w = w - ETA * w

            #print "after", w

    endtime = datetime.datetime.now()
    timeDiff = endtime - starttime

    ## Step 6: print resulting weight vector w
    np.savetxt('weightvector_mapper.txt', (w), newline=' ')

    innerProducts = []
    correct = 0
    total = len(x_all)
    for i in range(0, len(x_all)):
        innerProducts.append(np.inner(w, x_all[i]))
        if (label_all[i]*np.inner(w, x_all[i]) >= 0):
            correct = correct + 1
    print(w)
    print(np.abs(np.sum(w)))    
    print "time: ", timeDiff
    print "ETA: ", ETA
    print "b: ", b
    print "c: ", C
    print "total points: ", total
    print "correct classifications: ", correct
    print "accuracy: ", float(correct)/float(total)
    print "min inner product: ", min(innerProducts)
    print "max inner product: ", max(innerProducts)

## Additional Information
## http://scikit-learn.org/stable/modules/sgd.html#sgd
## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
## http://www.bogotobogo.com/python/python_numpy_batch_gradient_descent_algorithm.php
## python support vector machines stochastic gradient descent
## svm stochastic gradient descent python
## http://datascience.stackexchange.com/questions/1246/stochastic-gradient-descent-based-on-vector-operations
## http://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
## http://deeplearning.net/tutorial/gettingstarted.html
## https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent
