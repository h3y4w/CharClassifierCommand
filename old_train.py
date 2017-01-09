from nn import neuralNetwork
import numpy
import scipy
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

def _print_records(i, tdl, e, epochs):
    message = "{}/{} records ({}/{} epochs){}"
       
    if i/(e+1)%len(tdl)==0:


        print message.format(i-len(tdl)*e, len(tdl), e+1, epochs, " Done")

    #elif i%5000:

    elif i%(len(tdl)*.25)==0:

        print message.format(i-len(tdl)*e, len(tdl), e+1, epochs, "...")

def _train(fn, i_nodes, h_nodes, o_nodes, l_rate, epochs, out): #input, hidden, output

    # learning rate
    #l_rate = 0.1

    # create instance of neural network
    n = neuralNetwork(i_nodes,h_nodes,o_nodes, l_rate)

    #cell 5
    # load the mnist training data CSV file into a list

    print "Opening and extracting data...",
    training_data_file = open(fn, 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    print "OK"

    #cell 6
    # train the neural network

    # epochs is the number of times the training data set is used for training
 #   epochs = 1
    i=0
    print "Training with records in data set..."
    for e in range(epochs):
        # go through all records in the training data set
        for record in training_data_list:
            i+=1
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

            _print_records(i, training_data_list, e, epochs) 

    print "Saving data...",
    n.save(out)
    print "Ok"



