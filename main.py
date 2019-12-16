#REG-NO: COM/B/01-02204/2016
#NAME: ODERA DICKENS OCHIENG

import numpy as np #library for manipulating arrays

class XOR():
#definition of the constructor and initialization of the class variables (weights, inputs, biases)
    def __init__(self):
        self.x1 = 0
        self.x2 = 1
        self.w1 = 4.83
        self.w2 = -4.83
        self.w3 = -4.83
        self.w4 = 4.60 
        self.w5 = 5.73 
        self.w6 = 5.83
        self.b1 = -2.82
        self.b2 = -2.74
        self.b3 = -2.86
#the sigmoid activation function(gives probabilities ranging from 0-1)
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
        
#weighted summation of H1(node 1 on the first hidden layer of the neural network)
    def hidden_node_1(self):
        return (self.x1 * self.w1) + (self.x2 * self.w2) + self.b1

#activivation function of the first node (takes in weighted summation from node 1,i.e H1)
    def node_1_activation(self):
        return self.sigmoid(self.hidden_node_1())

#weighted summation of H2(node 2 on the first hidden layer of the neural network)
    def hidden_node_2(self):
        return (self.x1 * self.w3) + (self.x2 * self.w4) + self.b2

#activivation function of the second node (takes in weighted summation from node 2,i.e H2)
    def node_2_activation(self):
        return self.sigmoid(self.hidden_node_2())

#the weighted summation of the output node of the neural network
    def output_layer_summation(self):
        return (self.node_1_activation() * self.w5) + (self.node_2_activation() * self.w6) + self.b3

#the output of the neural network(sigmoid function value)
    def output(self):
        return self.sigmoid(self.output_layer_summation())
    
#An implementation of the perceptron
class Perceptron():
    #contructor with initialized inputs, weights and other perceptron parameters
    def __init__(self):
        self.inputs = [1,0,1]
        self.weights = [0.5,0.2,0.8]
        self.output_val = 1
        self.threshold = 0.05
        self.expected_val = 1 #or 0

#activation function        
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def weighted_summation(self):
        return np.dot(self.inputs, self.weights) #multiplies inputs by corresponding weights in the arrays

#a function to update the weights of the perceptron
    def update_weights(self):
        for rows in range(len(self.weights)):
            for columns in range(len(self.inputs)):
                new_weights = self.weights[rows] + (self.threshold * (self.expected_val - self.output_val) * self.inputs[columns]) 
                self.weights += new_weights
                return self.weights

#implementation of the threshold function(x = 1 if value greator than 1.2 or 0 otherwise)
    def output(self):
        res = self.sigmoid(self.weighted_summation())
        if res >= 1.2:
            return 1
        else:
            return 0

#main fuction used to call the objects of the above two classes
def main():
        neural_network = XOR()
        perceptron = Perceptron()

        print("XOR Neural Network Output")
        print(neural_network.output())

        print("Perceptron Implementation Output")
        print(perceptron.output())


#running the application
if __name__ == "__main__":
    main()