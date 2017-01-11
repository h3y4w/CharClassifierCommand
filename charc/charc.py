#!/usr/bin/env python
import sys
import argparse
from nn import NeuralNetwork
from termcolor import colored
from CharcExceptions import InputNodesException
import time

class CharClassifierCommand (object):
    prog = "charc"

    def __init__(self, cc):

        self.cc = cc

    def pair_args(self, args):
        items= []
        for i, item in enumerate(args):
            if not ((i+1)%2==0):
                try:
                    items.append([item, args[i+1]])
                except IndexError:
                    self.print_("ERROR: arg {} doesn't have a label.  It's been ignored".format(item), c="red", c_attrs=["bold"])

        return items


    def print_(self, message, trailing=False, c=None, c_attrs=[], use_prog=True):
        m = message
        if c is not None:
            m = colored(message, c, attrs=c_attrs)
        if use_prog:
            m = self.prog+': '+m
        if trailing:
            print m,

        else:
#            sys.stdout.write('')
            print m


    def do_convert(self, command):

        convert_parser = argparse.ArgumentParser(prog=self.prog)

        convert_parser.add_argument('-p', '--path', required=True, nargs="*", dest="paths")

        args = convert_parser.parse_args(command)
        
        for path in args.paths:
            print(cc.get_images(path))



    def do_test(self, command):

        test_parser = argparse.ArgumentParser(prog=self.prog)

        test_parser.add_argument('-d', '--data', required=True, dest="fn")
        test_parser.add_argument('-i', '--image', required=True, nargs="*") 


        
        args = test_parser.parse_args(command)
        images = self.pair_args(args.image)

        n = NeuralNetwork.load(args.fn)
        results = []

        self.print_("Starting image analysis...", c="magenta", c_attrs=["bold"])
        for i, image_info in enumerate(images):
            inputs = n.inputImage(image_info[0]) #image 7 & 4 don't work because they are not correct pixels
            self.print_('  Processing {} ({}/{})...'.format(image_info[0], i+1, len(images)), trailing=True, use_prog=False) 
            try:
                r = n.query(inputs)
            except InputNodesException:
                self.print_("ERROR: Input nodes are not equal to the pixels", c="red", c_attrs=["bold"], use_prog=False)
            else:
                self.print_(' OK', c="blue", c_attrs=["bold"], use_prog=False)
                results.append([image_info[0], image_info[1], r.label, r.elapsed_ms])
        print("")

        if len(results) !=0:
            self.print_("\tRESULTS", c="magenta", c_attrs=["bold"], use_prog=False)
            num_correct = 0
            elapsed_time = 0
            for i in xrange(0, len(results)):
                match = int(results[i][1]) == results[i][2]
                self.print_("\t  Image: {}".format(results[i][0]), use_prog=False)
                self.print_("\t  Label: {}".format(results[i][1]), use_prog=False)
                self.print_("\t  Response: {}".format(results[i][2]), use_prog=False)
                self.print_("\t  Match:", use_prog=False, trailing=True)

                elapsed_time+=results[i][3]
                if match:
                    self.print_("TRUE", c="green", c_attrs=["bold"], use_prog=False)
                    num_correct+=1
                else:
                    self.print_("FALSE", c="red", c_attrs=["bold"], use_prog=False)


                self.print_("\t  Elapsed (ms): {}".format(results[i][3]), use_prog=False)

                if len(results) > i+1:
                    print("") #newline

            print("")
            accuracy = (float(num_correct)/len(results))*100           
            self.print_("\tAccuracy: {}%".format(accuracy), c="magenta", c_attrs=["bold"], use_prog=False)
            self.print_("\tElapsed (ms): {}".format(elapsed_time), c="magenta", c_attrs=["bold"], use_prog=False)
        else:
            self.print_("\tNO RESULTS", c="magenta", c_attrs=["bold"], use_prog=False)

        #self.cc._test(args.fn, images)
        

    def do_train(self, command):
        train_parser = argparse.ArgumentParser(prog=self.prog)

        train_parser.add_argument('-d', '--data', type=str, required=True, dest="fn", nargs="*",
                                        help="path to the data file csv format")

        train_parser.add_argument('-i', '--inodes', type=int, default=784,
                                        help="Number of input nodes")

        train_parser.add_argument('-o', '--onodes', type=int, default=200,
                                        help="Number of output nodes")

        train_parser.add_argument('-n', '--hnodes', type=int, default=200,
                                        help="Number of hidden nodes")

        train_parser.add_argument('-l', '--l_rate', type=float, default=0.1,
                                        help="Learning rate")

        train_parser.add_argument('-e', '--epoch', type=int, default=5,
                                        help="Numbers of iteration data set will be used in training")

        train_parser.add_argument('--out', type=str, default="out")

        args = train_parser.parse_args(command)
        
        self.print_("Creating Network object..", c="magenta", c_attrs=["bold"], trailing=True)
        n = NeuralNetwork(args.inodes,args.hnodes,args.onodes, args.l_rate)
        self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)

        self.print_("Converting Images to arrays...", c="magenta", c_attrs=["bold"])
        data = []
        for i, csv in enumerate(args.fn):
            d1 = n.inputCSV(csv, training=True)
            data.append(d1)

        self.print_("Training Network...", c="magenta", c_attrs=["bold"])
        for j, item in enumerate(data):
            for i in xrange(0, len(item.inputs_list)):
                percent = str((float(i+1)/len(item.inputs_list))*100)[:5]
                data = n.train(item.inputs_list[i], item.targets_list[i], args.epoch)
                sys.stdout.flush()

                self.print_("\r\t{}%".format(percent), c="magenta", c_attrs=["bold"], use_prog=False, trailing=True)

        self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)


        self.print_("Saving Network ({}.pkl)...".format(args.out), c="magenta", c_attrs=["bold"], trailing=True)
        n.save(args.out)
        self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)


        #self.cc._train(args.fn, args.inodes, args.hnodes, args.onodes, args.l_rate, args.epoch, args.out)


    def do(self, command):
        if command[0] == "train":
            self.do_train(command[1:])

        elif command[0] == "test":
            self.do_test(command[1:])

        elif command[0] == "convert":
            self.do_convert(command[1:])

        else:
            print('usage: charc [train] [test]')
            print("{}: {} is an unknown command".format(self.prog, command[0]))


if __name__ == "__main__":

    if len(sys.argv)<2:
        print('usage: charc [train] [test]')
        exit(1)
    
    ccc = CharClassifierCommand('test')
    ccc.do(sys.argv[1:])




