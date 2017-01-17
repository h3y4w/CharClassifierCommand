import numpy
import math
from datetime import datetime
import pickle
#from cloud.serialization.cloudpickle import dumps
import cloudpickle
import dill
import hickle
import scipy.special
from CharcExceptions import InputNodesException

class Namespace (object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

    def __repr__(self):
        return str(self.__dict__.keys())

class Record (object):
    def __init__(self, inputs=None, input_nodes=None, output_nodes=None, targets=None, label=None):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.inputs = inputs
        self.targets = targets
        self.label = label

    def input_image(self, fn, label=None):
        img_array = scipy.misc.imread(fn, flatten=True)
        pixels = img_array.shape[0]*img_array.shape[1]

        img_data = None
        try:
            img_data = 255.0 - img_array.reshape(pixels)
        except ValueError:
            raise InputNodesException("Pixels do not match number of input nodes")

        img_data = (img_data / 255.0 * 0.99) + 0.01
        #raw_data = []
        #for v in img_data:
        #    _v = v
        #    if v!=0:
        #        _v = 1
        #    raw_data.append(_v)
        #img_data = raw_data

        self.inputs = img_data
        self.input_nodes = pixels
        self.targets = numpy.zeros(self.output_nodes) + 0.01
        self.label = label

    def training_data(self):
        return numpy.append(label, img_data)

    def store_data(self):
        return self.__dict__

    @staticmethod
    def load(stored_record):
        inputs = stored_record['inputs']
        input_nodes = stored_record['input_nodes']
        targets = stored_record['targets']
        label = stored_record['label']

        return Record(inputs=inputs, input_nodes=input_nodes, targets=targets, label=label)

    def save_image(self, fn):
        #saves record as an image
        pass

    def save_csv(self, fn):
        #saves record as a csv file
        pass


def _convert_csv(values, output_nodes, training=False):
    targets = None
    index = 0 #the first index is not a label so it assumes their is no label
    if training:
        index = 1 #index 0 is the label, so extract data from everything after that
        label = values[0]

        #_values = []
        #for v in values[index:]:
        #    if v !=str(0):
        #        v=.99
        #    _values.append(v)

        # all_values[0] is the target label for this record
        #labels are all ascii values.  Num from 0..9 are not ascii rather
        try:
            label = int(label)

        except ValueError:
            label = ord(label)

    inputs = (numpy.asfarray(values[index:]) / 255.0 * 0.99) + 0.01 #normalize data by dividing by 255

    targets = numpy.zeros(output_nodes) + 0.01

    if label is not None:
        targets[label] = 1

    return inputs, targets, label

class Dataset (object):
    def __init__(self, input_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.records = []
        self.stored_records = []
        self.training = None

    def input_csv(self, fn, training=False):
        if not (self.training == None or self.training == training):
            raise ValueError("Datasets can't merge.  Dataset is already {}".format(self.training))
        self.training = training
        data_list = None
        with open(fn, 'r') as f:
            #loops through csv data and checks for empty lines then appends it to data
            data_list = [line.replace("\n", "") for line in f.readlines() if line!="" and line!="\n"]
        total_elapse = datetime.now()

        # go through all records in the training data set
        for record in data_list:
            # split the record by the ',' commas
            all_values = record.split(',')

            inputs, targets, label = _convert_csv(all_values, self.output_nodes, training=training)

            r = Record(inputs=inputs, targets=targets, input_nodes=self.input_nodes, output_nodes=self.output_nodes, label=label)
            self.records.append(r)
        return self.records

    def output_csv(self, fn, training=False):
        pass

    def add_record(self, record):
        if record.input_nodes == self.input_nodes:
            self.records.append(record)
        else:
            raise InputNodesException("Record's input nodes doesn't match Dataset's input nodes")
    def get_record(self, index):
        try:
            return self.records[index]
        except ValueError:
            pass

    def save(self, fn):
        self.stored_records = []
        for record in self.records:
            self.stored_records.append(record.store_data())
        #hickle.dump(self, fn+'.dataset.pkl', mode='w')
        with open(fn+".dataset.pkl", "wb") as output_:
            #output_.write(cloudpickle.dumps(self))
            dill.dump(self, output_)

    @staticmethod
    def load(fn):
        dataset = None
        #dataset = hickle.load(fn)
        with open(fn, "rb") as input_:
            #return pickle.load(input_)
            dataset = dill.load(input_)
        for stored_record in dataset.stored_records:
            dataset.add_record(Record.load(stored_record))

        del dataset.stored_records

        return dataset


class Sample (object):
    pass

class NeuralNetwork (object):

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes
        self.lr = learningrate
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # learning rate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)


    def input_csv(self, fn, training=False):
        data_list = None
        inputs_list = []
        targets_list = []
        with open(fn, 'r') as f:
            #loops through csv data and checks for empty lines then appends it to data
            data_list = [line.replace("\n", "") for line in f.readlines() if line!="" and line!="\n"]


        total_elapse = datetime.now()

        # go through all records in the training data set
        records = []
        for record in data_list:
            # split the record by the ',' commas
            all_values = record.split(',')

            inputs, targets, label = convert_csv(all_values, self.output_nodes, training=training)

            r = Record(label, inputs, targets)
            records.append(r)

            inputs_list.append(inputs)
            targets_list.append(targets)

        total_elapse = ((datetime.now()-total_elapse)*1000).total_seconds()

        return Namespace({'inputs_list':inputs_list, 'targets_list':targets_list, 'elapsed_ms':total_elapse})


    # train the neural network
    def train(self, record, epochs):

        start = datetime.now()

        #FOR THE TIMEBING WHILE TESTING WHETHER TO LOOP THE DATA WITHIN HERE
        for e in xrange(0, epochs):

        # convert inputs list to 2d array
            inputs = numpy.array(record.inputs, ndmin=2).T
            targets = numpy.array(record.targets, ndmin=2).T

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
    def query(self, record):
        start = datetime.now()

        # convert inputs list to 2d array
        inputs = numpy.array(record.inputs, ndmin=2).T
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
        with open(fn+".nn.pkl", "wb") as output_:
            #output_.write(cloudpickle.dumps(self))
            dill.dump(self, output_)


    @staticmethod
    def load(fn):
        with open(fn, "rb") as input_:
            #return pickle.load(input_)
            return dill.load(input_)


