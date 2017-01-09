import scipy.misc
import numpy
from nn import neuralNetwork
from subprocess import check_output 
import os
import sys
import operator
import functools


_PIXELS = 784
_VERBOSE = True 

def print_v(message, trailing=False):
    if _VERBOSE:
        if trailing:
            print message,
        else:
            print message

def _print_lines(message_length, char):
    for x in xrange(0, message_length/2):
        if x+1<message_length/2:
            print char,
        else:
            print char 

def _test(fn, image_files):
    n = neuralNetwork.load(fn)

    dataset = []
    print_v("Analyzing Data...")
    i=0
    num_correct = 0

    for image_info in image_files:
        image_file_name = image_info.split(":")[0]
        label = int(image_info.split(":")[1])

        i+=1
        print_v("Processing {}... ({}/{})".format(image_file_name, i, len(image_files)),trailing=True)

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
            print_v("ERROR: needs to have 7:7 Ratio".format(image_file_name))
            continue

        r_message = " DONE"
        if (_PIXELS != pixels):
            converted_image = "{}.resized28x28".format(image_file_name)

            r1 = check_output("convert {} -resize 28x28 {}".format(image_file_name, converted_image).split(" "))
            if (len(r1)>1):
                print_v("ERROR: could not be converted".format(image_file_name))
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
        record = numpy.append(label,img_data)
        dataset.append(record)

        ender = "..."
        if i==len(image_files):
            ender = " Done"
        #print "    Processing {}/{}{}".format(i, len(image_files), ender)
        print_v(r_message)
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
        print_v("\nRESULTS")
        print_v("-------")
        print_v("LABEL: {}".format(correct_label))
        print_v("GUESS: {}".format(label))
        print_v("GUESS 2: {}".format(second_label))
        print_v("MATCH: {}".format(correct_label==label))
        if correct_label==label:
            num_correct+=1
        extracted_data.append(d)

    print_v("-------------------")
    print_v("Accuracy: {}%".format((float(num_correct)/len(dataset))*100))
    return extracted_data

        #print "{} --->".format(image_files[item]),
        #print "{} && {} <----- network ({})".format(correct_label, label, correct_label==label)
        # append correct or incorrect to list
