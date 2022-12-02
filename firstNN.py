import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm  # a library that shows a progress bar
import matplotlib.pyplot as plt


def main():
    X_train, X_test, y_train, y_test = load_data()
    n_hidden, n_classes = 4, 2
    accuracy_list, loss_list = train_feddForward(X_train, X_test, y_train, y_test, n_hidden, n_classes)
    plot_result(accuracy_list, loss_list)


def load_data():
    data = pd.read_csv('Top 50 Spam Stems.csv', encoding= "utf-8")
    columnNames = []
    for col in data.columns:
        if col == "Class":
            break
        else:
            columnNames.append(col)

    X = data[columnNames]
    y = data["Class"]


    # Scale data to have mean 0 and variance 1
    # which is importance for convergence of the neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data set into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=2)
    return X_train, X_test, y_train, y_test  # a tuple of the four data structures


class Model(nn.Module):

    def __init__(self, n_features, n_hidden, n_classes):  # constructor of the class takes the input size as a param
        super(Model, self).__init__()  # we call the constructor of the base class
        self.layer1 = nn.Linear(n_features, n_hidden)  # input layer, input=n_features, output=n_hidden
        self.layer2 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.layer3 = nn.Linear(n_hidden, n_classes)  # output layer, we have 3 classes in the IRIS dataset

    def forward(self, x):  # we set the activation function
        x = F.relu(self.layer1(x))  # relu definition is f(x)=max(0, x)
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x),
                      dim=1)  # softmax turns a vector of K real values into a vector of K real values that sum to 1
        return x


def train_feddForward(X_train, X_test, y_train, y_test, n_hidden, n_classes):
    ''' '''

    model = Model(X_train.shape[1], n_hidden, n_classes)  # we pass the trainning part of the iris data

    ''' 
        The Adam optimization algorithm is an extension to stochastic gradient descent 
        that has recently seen broader adoption for deep learning applications 
        in computer vision and natural language processing.
        It is used to update network weights based in training data.
    '''

    # optimizer =  torch.optim.SGD(model.parameters(), lr = 0.01) # stochastic gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # learning rate is 0.001

    '''
       CrossEntropyLoss()
       It is useful when training a classification problem with C classes. 
       If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. 
       This is particularly useful when you have an unbalanced training set.
    '''

    loss_fn = nn.CrossEntropyLoss()

    print(model)

    EPOCHS = 500
    # we convert the data from numpy to torch
    # Series.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print(torch.from_numpy(y_train))
    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).long()

    # we initialize the loss and accuracy lists to zero
    loss_list = np.zeros((EPOCHS,))
    accuracy_list = np.zeros((EPOCHS,))

    for epoch in tqdm.trange(EPOCHS):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)  # loss function
        loss_list[epoch] = loss.item()  # we append the loss to the list of loss

        ''' 
         Zero gradients
         we need to set the gradients to zero 
         before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        '''

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # that updates the parameters

        ''' 
        Context-manager that disabled gradient calculation.

        Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). 
        It will reduce memory consumption for computations that would otherwise have requires_grad=True.
        '''

        with torch.no_grad():
            y_pred = model(X_test)
            correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    return accuracy_list, loss_list


def plot_result(accuracy_list, loss_list):
    ''' plots both the accuracy and the loss given the number of epchos '''

    print("plotting the results ...")
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

    ax1.plot(accuracy_list)
    ax1.set_ylabel("validation accuracy")
    ax2.plot(loss_list)
    ax2.set_ylabel("validation loss")
    ax2.set_xlabel("epochs")
    plt.show()


main()