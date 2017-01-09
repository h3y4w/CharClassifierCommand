#!/usr/bin/env python

from resources import CharClassifier 
import sys
import argparse

class CharClassifierCommand (object):
    prog = "charc"

    def __init__(self, cc):
        self.cc = cc

    def do_test(self, command):

        test_parser = argparse.ArgumentParser(prog=self.prog)

        test_parser.add_argument('-d', '--data', required=True, dest="fn")
        test_parser.add_argument('-i', '--images', required=True, nargs="*") 
        
        args = test_parser.parse_args(command)
        self.cc._test(args.fn, args.images)
        

    def do_train(self, command):
        train_parser = argparse.ArgumentParser(prog=self.prog)

        train_parser.add_argument('-d', '--data', type=str, required=True, dest="fn",
                                        help="path to the data file csv format")

        train_parser.add_argument('-i', '--inodes', type=int, default=784,
                                        help="Number of input nodes")

        train_parser.add_argument('-o', '--onodes', type=int, default=10,
                                        help="Number of output nodes")

        train_parser.add_argument('-n', '--hnodes', type=int, default=200,
                                        help="Number of hidden nodes")

        train_parser.add_argument('-l', '--l_rate', type=float, default=0.1,
                                        help="Learning rate")

        train_parser.add_argument('-e', '--epoch', type=int, default=5,
                                        help="Numbers of iteration data set will be used in training")

        train_parser.add_argument('--out', type=str, default="out")

        args = train_parser.parse_args(command)
        
        
        self.cc._train(args.fn, args.inodes, args.hnodes, args.onodes, args.l_rate, args.epoch, args.out)


    def do(self, command):
        if command[0] == "train":
            self.do_train(command[1:])

        elif command[0] == "test":
            self.do_test(command[1:])
        else:
            print 'usage: charc [train] [test]'
            print "{}: {} is unknown".format(self.prog, command[0])


if __name__ == "__main__":


    if len(sys.argv)<2:
        print 'usage: charc [train] [test]'
        exit(1)
    
    cc = CharClassifier(verbose=True, prog=CharClassifierCommand.prog+": ")
    ccc = CharClassifierCommand(cc)
    ccc.do(sys.argv[1:])




