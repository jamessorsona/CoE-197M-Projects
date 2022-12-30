""" 
James Carlo E. Sorsona
2019-01401
CoE 197 M-THY
Assignment 2: Polynomial Solver using SGD in TinyGrad

SGD is a useful algorithm with many applications. In this assignment, we will use SGD in the TinyGrad framework as polynomial solver - to find the degree and coefficients.

The solver will use data_train.csv to estimate the degree and coefficients of a polynomial. To test the generalization of the learned function, it should have small test error on data_test.csv.
The function should be modeled using tinygrad : https://github.com/geohot/tinygrad
Use SGD to learned the polynomial coefficients.
"""

# tinygrad module
from tinygrad.tensor import Tensor
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

# other modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from operator import itemgetter

# ML Model
class Model:
    def __init__(self, pol_deg):
        self.poly = [Tensor([np.array([np.random.uniform(-1,1)])], requires_grad=True) for i in range(pol_deg+1)]

    def load_polynomial(self, polynomial):
        self.poly = [Tensor([np.array(coeff)], requires_grad = True) for coeff in polynomial]

    def reshaped_tensor(self):
        reshaped_tensor = self.poly[0]
        [reshaped_tensor := reshaped_tensor.cat(self.poly[i]) for i in range(1,len(self.poly))]
        return reshaped_tensor

    def forward_pass(self,x):
        return x.matmul(self.reshaped_tensor()).sum(axis=1)  

    def get_polynomial(self):
        return [i.data[0] for i in self.poly]

class Load_Model:
    def __init__(self, x, y, batch_size=32, shuffle=True, polyfeatures=True, degree=4):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.polyfeatures = polyfeatures
        self.degree = degree
        self.shuffle = shuffle

    def __iter__(self): 
        # splits an iterable in constant size chunks
        if self.shuffle: 
            self.shuffle_data()
        for i in range(0, len(self.x),self.batch_size):
            x_batch = self.x[i:min(i + self.batch_size, len(self.x))]
            y_batch = self.y[i:min(i + self.batch_size, len(self.x))]
            if self.polyfeatures: 
                x_batch = self.get_features(x_batch)
            yield x_batch, y_batch
            

    def get_features(self, x):
        b = np.hstack((np.ones((len(x), 1)),x))
        feats = np.hstack([((b[:,1] ** i).reshape((len(x),1))) for i in range(self.degree+1)])

        tensor_features = Tensor(feats, requires_grad = False)
        return tensor_features

    def unshuffle_data(self):
        unshuffle = sorted(zip(self.x,self.y), key=itemgetter(0))
        self.x, self.y = zip(*unshuffle)

    def shuffle_data(self):
        x_shuffled = []
        y_shuffled = []

        i = np.random.permutation(len(self.x))
        for idx in range(len(self.x)):
            x_shuffled.insert(i[idx],self.x[idx])
            y_shuffled.insert(i[idx],self.y[idx])
        self.x = x_shuffled
        self.y = y_shuffled

def get_mean_squared_error(y, y_hat):
    return ((y-y_hat)**2).mean()

def main():

    # import data
    data_train = pd.read_csv('./data_train.csv')
    data_test = pd.read_csv('./data_test.csv')

    # split the training and test data
    x_train, x_testing, y_train, y_testing = train_test_split([[i] for i in data_train["x"]], [[i] for i in data_train["y"]], test_size=0.20, random_state=42)
    x_test, y_test = [[i] for i in data_test["x"]], [[i] for i in data_test["y"]]

    # hyperparameters
    epochs = 200
    batch_size = 32
    learning_rate = [3e-4, 3e-5, 3e-6, 3e-8, 3e-10]
    
    best_models = []
    losses = []

    for degree in range(4,0,-1):

        print(f"Training model with degree {degree}...")

        # initialize model
        model = Model(degree)
        optimizer = optim.SGD(model.poly, lr=learning_rate[degree])
        train = Load_Model(x_train, y_train, batch_size, True, True, degree)
        testing = Load_Model(x_testing, y_testing, batch_size, False, True, degree)
        alpha = 0.5

        best = []
        best_loss = None

        # train model
        for epoch in range(epochs):

            for x, y in train:
                out = model.forward_pass(x).reshape(-1,1)
                l2_pen = alpha * model.reshaped_tensor()[1:].mul(model.reshaped_tensor()[1:]).sqrt().sum()
                loss = get_mean_squared_error(y, out) + l2_pen
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            validate_loss = 0.0

            # validate model
            for x, y in testing:
                out = model.forward_pass(x).reshape(-1,1)
                vloss = get_mean_squared_error(y, out).data[0]
                validate_loss += vloss*len(out.data[0])
            
            # normalize the loss
            validate_loss /= len(train.x)

            # save the best model
            best_loss = validate_loss
            if best_loss is not None and best_loss > validate_loss:
                best = [c[0] for c in model.get_polynomial()]
                    

            best_models.append([c[0] for c in model.get_polynomial()])
            losses.append(best_loss)

    # get the best model
    best_coeffs = best_models[losses.index(min(losses))]

    print(f"Degree: {len(best_coeffs)}")
    print(f"Coefficients: {best_coeffs}")

    # testing the best model against data_test
    best_model = Model(len(best_coeffs)-1)
    best_model.load_polynomial(np.array(best_coeffs).reshape(-1,1))

    test = Load_Model(x_test,y_test,64, False, True, len(best_coeffs)-1)

    y_test_pred = best_model.forward_pass(test.get_features(x_test)).reshape(-1, 1)
    test_r2 = metrics.r2_score(y_test, y_test_pred.data)

    print(f"R2 Score: {test_r2}")


if __name__ == '__main__':
    main()