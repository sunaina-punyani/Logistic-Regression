
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA =1.8                                                                                       
EPOCHS = 5000#keep this greater than or equal to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    #make a column vector of ones
    n=X.shape[0]
    colA=np.ones((X.shape[0],1))
    newX=np.hstack((colA,X))
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #remove this line once you finish writing
    return newX



 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
	return np.zeros(n_thetas)	



def train(theta, X, y, model):
     
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
     for i in range (0,EPOCHS):
     	y_p=predict(X,theta)
     	cost=costFunc(m,y,y_p)
     	J.append(cost)
     	grad=calcGradients(X,y,y_p,m)
     	theta=makeGradientUpdate(theta,grad)

    
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)

     model['J'] = J
     model['theta'] = list(theta)
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    sub=np.subtract(y_predicted,y)
    prod=np.multiply(sub,sub)
    j=np.sum(prod)
    j=j/(2*m)
    return j

def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # basically an numpy array containing n_params
    b=np.subtract(y_predicted,y)
    
    n2=b.shape[0]
    b=b.reshape((n2,1))
    a=X*b
    c=np.sum(a,axis=0)
    d=c/m
    return d

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    al=ALPHA*grads
    theta=np.subtract(theta,al)
    return theta


#this function will take two paramets as the input
def predict(X,theta):
    y=np.dot(X,theta)
    y=np.exp(-y)
    y=y+1
    y=1/y
    return y


########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
       # with open(MODEL_FILE,'w') as f:
        #    f.write(json.dumps(model))
        print("Training Data:")
        accuracy(X,y,model)

    #else:
     #   model = {}
        #with open(MODEL_FILE,'r') as f:
           # model = json.loads(f.read())
        X_df, y_df = loadData(FILE_NAME_TEST)
        X,y = normalizeTestData(X_df, y_df, model)
        X = appendIntercept(X)
        print("Testing Data:")
        accuracy(X,y,model)
        #with open(MODEL_FILE,'w') as f:
         #   f.write(json.dumps(model))

if __name__ == '__main__':
    main()
