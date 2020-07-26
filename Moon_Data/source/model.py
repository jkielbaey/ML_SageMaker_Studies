import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        print("Making SimpleNet with input {}, hidden {}, output {}".format(input_dim, hidden_dim, output_dim))
        
        # defining 2 linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # defining dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # defining sigmod layer
        self.sig = nn.Sigmoid()
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return self.sig(out)
