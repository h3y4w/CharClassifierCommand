import numpy
import math
from datetime import datetime
import pickle
#from cloud.serialization.cloudpickle import dumps
import cloudpickle
import scipy.special
from CharcExceptions import InputNodesException

class Namespace (object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        return str(self.__dict__.keys())

class NeuralNetwork (object):
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass


    def inputCSV(self, fn, training=False):
        data_list = None
        inputs_list = [] 
        targets_list = []
        with open(fn, 'r') as f:
            #loops through csv data and checks for empty lines then appends it to data
            data_list = [line.replace("\n", "") for line in f.readlines() if line!="" and line!="\n"]


        total_elapse = datetime.now()

        # go through all records in the training data set
        for record in data_list:
            # split the record by the ',' commas
            all_values = record.split(',')



            index = 0 #the first index is not a label so it assumes their is no label
            if training:
                index = 1 #index 0 is the label, so extract data from everything after that
                label = all_values[0]

                # all_values[0] is the target label for this record

                #labels are all ascii values.  Num from 0..9 are not ascii rather
                try:
                    label = int(label)

                except ValueError:
                    label = ord(label) 

            inputs = (numpy.asfarray(all_values[index:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(self.onodes) + 0.01

            if label is not None:
                targets[label] = 0.99
            
            inputs_list.append(inputs)
            targets_list.append(targets)

        total_elapse = ((datetime.now()-total_elapse)*1000).total_seconds()

        return Namespace({'inputs_list':inputs_list, 'targets_list':targets_list, 'elapsed_ms':total_elapse})

    def inputImage(self, fn, training=False, label=None):
        img_array = scipy.misc.imread(fn, flatten=True)
        pixels = img_array.shape[0]*img_array.shape[1]

        img_data = None
        try:
            img_data = 255.0 - img_array.reshape(pixels)
        except ValueError:
            raise InputNodesException("Pixels do not match number of input nodes")

        img_data = (img_data / 255.0 * 0.99) + 0.01
        if training == True and label is not None:
            img_data = numpy.append(label, img_data)

        return img_data



    # train the neural network
    def train(self, inputs_list, targets_list, epochs):

        start = datetime.now()

        #FOR THE TIMEBING WHILE TESTING WHETHER TO LOOP THE DATA WITHIN HERE
        i_l = inputs_list
        t_l = targets_list
        inputs_list = [i_l]
        targets_list = [t_l]
        i = 0

        for e in xrange(0, epochs): 

            #for i in xrange(0, len(inputs_list)):
            if True: #timebeing while testing
            
            # convert inputs list to 2d array
                inputs = numpy.array(inputs_list[i], ndmin=2).T
                targets = numpy.array(targets_list[i], ndmin=2).T
                
                # calculate signals into hidden layer
                hidden_inputs = numpy.dot(self.wih, inputs)
                # calculate the signals emerging from hidden layer
                hidden_outputs = self.activation_function(hidden_inputs)
                
                # calculate signals into final output layer
                final_inputs = numpy.dot(self.who, hidden_outputs)
                # calculate the signals emerging from final output layer
                final_outputs = self.activation_function(final_inputs)
                
                # output layer error is the (target - actual)
                output_errors = targets - final_outputs
                # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
                hidden_errors = numpy.dot(self.who.T, output_errors) 
                
                # update the weights for the links between the hidden and output layers
                self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        elapsed_ms = ((datetime.now() - start)*1000).total_seconds()
        return Namespace({'output_errors':output_errors, 'elapsed_ms':elapsed_ms}) #returns errors and elapsed time

    # query the neural network
    def query(self, inputs_list):
        start = datetime.now()

        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T


        # calculate signals into hidden layer
        try:
            hidden_inputs = numpy.dot(self.wih, inputs)
        except ValueError:
            raise InputNodesException("Number of pixels is not equal to the number of input nodes")

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        elapsed_ms = ((datetime.now() - start)*1000).total_seconds()
        
        return Namespace({'outputs':final_outputs, 'elapsed_ms':elapsed_ms, 'label':numpy.argmax(final_outputs)})


        
    def save(self, fn):
        with open(fn+".pkl", "wb") as output_:
            output_.write(cloudpickle.dumps(self))

    @staticmethod
    def load(fn):
        with open(fn, "rb") as input_:
            return pickle.load(input_)




