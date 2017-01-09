from nn import neuralNetwork
import numpy
import scipy

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

import scipy.misc
import numpy
from nn import neuralNetwork
from subprocess import check_output 
import os
import sys
import operator
import functools
from termcolor import colored 

_PIXELS = 784


class CharClassifier (object):
    def __init__(self, verbose=False, prog=""):
        self.verbose = verbose
        self.prog = prog

    def print_v(self, message, trailing=False, c=None, c_attrs=[], use_prog=True):
        if self.verbose:
            m = message
            if c is not None:
                m = colored(message, c, attrs=c_attrs)
            if use_prog:
                m = self.prog+m
            if trailing:
                print m,
            else:
                print m

    def _print_records(self, i, tdl, e, epochs):
        if self.verbose:
            message = "    {}/{} records ({}/{} epochs)"
               

            ratio_i = i-len(tdl)*e #finds lowest numerator of i/tdl

            if (ratio_i == len(tdl) and e+1==epochs):
                print message.format(ratio_i, len(tdl), e+1, epochs) + colored(" DONE", "green", attrs=["bold"])

            elif i/(e+1)%len(tdl)==0:


                print message.format(ratio_i, len(tdl), e+1, epochs) + colored(" OK", "blue", attrs=["bold"])

            #elif i%5000:

            elif i%(len(tdl)*.25)==0:

                print message.format(ratio_i, len(tdl), e+1, epochs) + "..."

    def _print_lines(message_length, char):
        if self.verbose:
            for x in xrange(0, message_length/2):
                if x+1<message_length/2:
                    print char,
                else:
                    print char 


    def _test(self, fn, image_files):
        n = neuralNetwork.load(fn)

        dataset = []
        self.print_v("Analyzing Data...", c="magenta", c_attrs=["bold"])
        i=0
        num_correct = 0

        for image_info in image_files:
            image_file_name = image_info.split(":")[0]
            label = int(image_info.split(":")[1])

            i+=1
            self.print_v("    Processing {}... ({}/{})".format(image_file_name, i, len(image_files)),trailing=True, use_prog=False)

        #glob.glob('my_own_images/2828_my_own_*.png'):
           
            
            pixels = None

            if sys.platform == "linux" or sys.platform == "linux2":
                command = "identify -format '%w-%h' {}".format(os.path.join(os.getcwd(), image_file_name)).split(" ")
                pp = check_output(command).replace("\n", "").replace("'", "").split("-")

                if len(pp)==2:
                    pixels = int(pp[0])*int(pp[1])

                #img_size= r.split(" ")[2].split("x")
                #img_pixels = int(img_size[0])*int(img_size[1])

            else:
                from PIL import Image
                img = Image.open(image_file_name)
                pixels  = img.size[0]*img.size[1]

            if not (pixels % 784==0):
                self.print_v("ERROR: needs to have 7:7 Ratio".format(image_file_name), c="red", c_attrs=["bold"], use_prog=False)
                continue

            r_message = " DONE"
            if (_PIXELS != pixels):
                converted_image = "{}.resized28x28".format(image_file_name)

                r1 = check_output("convert {} -resize 28x28 {}".format(image_file_name, converted_image).split(" "))
                if (len(r1)>1):
                    self.print_v("ERROR: could not be converted".format(image_file_name), c="red", c_attrs=["bold"], use_prog=False)
                    continue
                else:
                    r_message += " (converted_image to 24 pixels)"
                    image_file_name = converted_image


            # use the filename to set the correct label
            #label = int(image_file_name[-5:-4])
            # load image data from png files into an array
            #print ("loading ... ", image_file_name)
            
            img_array = scipy.misc.imread(image_file_name, flatten=True)
            
            # reshape from 28x28 to list of 784 values, invert values
            
            #get relative path
            command = "identify -format '%w-%h' {}".format(os.path.join(os.getcwd(), image_file_name)).split(" ")
            pp = check_output(command).replace("\n", "").replace("'", "").split("-")
            pixels = None
            if len(pp)==2:
                pixels = int(pp[0])*int(pp[1])
                

            #pixels = 784

            img_data  = 255.0 - img_array.reshape(pixels)

            # then scale data to range from 0.01 to 1.0
            img_data = (img_data / 255.0 * 0.99) + 0.01
            #print(numpy.min(img_data))
            #print(numpy.max(img_data))
            
            # append label and image data  to test data set



            #print(numpy.min(img_data))
            #print(numpy.max(img_data))
            
            # append label and image data  to test data set
            record = numpy.append(label,img_data)
            dataset.append(record)

            ender = "..."
            if i==len(image_files):
                ender = " Done"
            #print "    Processing {}/{}{}".format(i, len(image_files), ender)
            self.print_v(r_message, c="green", c_attrs=["bold"])
            pass

        #cell 8

     

        extracted_data = []
        for item in xrange(0, len(dataset)):
            # record to test
            # correct answer is first value
            correct_label = dataset[item][0]
            second_label = dataset[item][2]

            # data is remaining values
            inputs = dataset[item][1:]
            # query the network
            outputs = n.query(inputs)
            #print (outputs)

            # the index of the highest value corresponds to the label
            d = {"label": label, "correct": correct_label, "match": correct_label==label}
            label = numpy.argmax(outputs)
            self.print_v("\nRESULTS")
            self.print_v("-------")
            self.print_v("LABEL: {}".format(correct_label))
            self.print_v("GUESS: {}".format(label))
            self.print_v("GUESS 2: {}".format(second_label))
            self.print_v("MATCH: {}".format(correct_label==label))
            if correct_label==label:
                num_correct+=1
            extracted_data.append(d)

        self.print_v("-------------------")
        self.print_v("Accuracy: {}%".format((float(num_correct)/len(dataset))*100))
        return extracted_data




    def _train(self, fn, i_nodes, h_nodes, o_nodes, l_rate, epochs, out): #input, hidden, output

        # learning rate
        #l_rate = 0.1

        # create instance of neural network
        n = neuralNetwork(i_nodes,h_nodes,o_nodes, l_rate)

        #cell 5
        # load the mnist training data CSV file into a list

        self.print_v("Opening and extracting data...", trailing=True, c="magenta", c_attrs=["bold"])
        training_data_file = open(fn, 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()
        self.print_v("OK", c="blue", c_attrs=["bold"], use_prog=False)

        #cell 6
        # train the neural network

        # epochs is the number of times the training data set is used for training
     #   epochs = 1
        i=0
        self.print_v("Training with records in data set...", c="magenta", c_attrs=["bold"])
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

                self._print_records(i, training_data_list, e, epochs) 

        self.print_v("Saving data...", trailing=True, c="magenta", c_attrs=["bold"])
        try:
            n.save(out)
            self.print_v("OK", c="blue", c_attrs=["bold"], use_prog=False)
        except Exception as e:
            self.print_v("ERROR: ({})".format(str(e)), c="red", c_attrs=["bold"], use_prog=False)

                
