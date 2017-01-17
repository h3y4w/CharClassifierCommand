#!/usr/bin/env python
import sys
import argparse
from nn import NeuralNetwork, Dataset, Record
from termcolor import colored
from CharcExceptions import InputNodesException
import time
import glob
import os
import math
from subprocess import check_output
from shutil import rmtree

class CharClassifierCommand (object):
    prog = "charc"
    do_map = dict(test="do_test", train="do_train", dataset="do_dataset")
    usage ='''charc <command> [<args>]
    
    Commands:
        train   Create or train existing neural network
        test    Test existing neural network
        dataset Create or update existing dataset'''

    def __init__(self):
        pass

    def pair_args(self, args):
        items= []
        for i, item in enumerate(args):
            if not ((i+1)%2==0):
                try:
                    items.append([item, args[i+1]])
                except IndexError:
                    self.print_("ERROR: arg {} doesn't have a label.  It's been ignored".format(item), c="red", c_attrs=["bold"])

        return items


    def print_(self, message, trailing=False, c=None, c_attrs=[], use_prog=True, raw_string=False):
        m = message
        if c is not None:
            m = colored(message, c, attrs=c_attrs)
        if use_prog:
            m = self.prog+": "+m
        if raw_string:
            m = "\r"+m
        if trailing:
            print m,

        else:
#            sys.stdout.write('')
            print m


    def do_dataset(self, command):
        tmp_dir = ".tmp_dataset_images"

        sample_parser = argparse.ArgumentParser(prog=self.prog)

        sample_parser.add_argument('-p', '--path', required=True, nargs="*", dest="paths")
        sample_parser.add_argument('-t', '--type', required=True, choices=['csv', 'image'])
        sample_parser.add_argument('-s', '--set', required=True, choices=['training', 'testing'])
        sample_parser.add_argument('-d', '--data', default=None)
        sample_parser.add_argument('--labelf', default=None, help="Filename format to extract label. EX. img_0_file.png img_1_file.png ... -> img_{label}_*.png")
        sample_parser.add_argument('-i', '--inodes', type=int)
        sample_parser.add_argument('-o', '--onodes', type=int)
        sample_parser.add_argument('--out', default="out")

        args = sample_parser.parse_args(command)

        if args.type == "image" and args.labelf == None:
            print "{}: error: If data type is image, argument --labelf must be set".format(self.prog)
            exit(1)

        elif (args.type =="image" and args.labelf.find('{label}')==-1):
            print '{}: error: Label is not set in arg --labelf'.format(self.prog)
            exit(1)


        data = []
        dataset = None
        if args.data is not None:
            dataset = Dataset.load(args.data)
        else:
            if args.inodes and args.onodes:
                dataset = Dataset(args.inodes, args.onodes)
            else:
                self.print_("ERROR: Input and ouput nodes needed", c="red", c_attrs=["bold"])
                exit(1)

        if args.type == "image":
            placeholder_length = len("{label}")
            labelf = args.labelf
            index = labelf.find("{label}")
            start = labelf[:index]
            end = labelf[index+placeholder_length:]
            #print 'label:{}'.format("img001-002.png"[index])

            for i, path in enumerate(args.paths):
                #print fn.find(start)
                files = glob.glob(path)
                for j, fn in enumerate(files):

                    label = fn[index+len(os.path.dirname(fn))+1]
                    try:
                        label = int(label)
                    except: #if label is not int, it will get ASCII VALUE AND USE THAT FOR THE LABEL
                        label = ord(label)

                    record = Record()

                    try:
                        record.input_image(fn, label=label)
                        dataset.add_record(record)
                    except InputNodesException:
                        if not os.path.exists(tmp_dir):
                            os.mkdir(tmp_dir)
                        n_fn = os.path.join(tmp_dir, os.path.basename(fn))
                        pixels_ = int(math.sqrt(dataset.input_nodes))
                        convert_command = "convert {0} -resize {1}x{1} {2}".format(fn, pixels_, n_fn).split(" ")
                        resp = check_output(convert_command)
                        if  (len(resp)>1):
                            self.print_("Error: Image {} could not be converted".format(fn), c="red", c_attrs=["bold"])
                            exit(1)
                        else:
                            record.input_image(n_fn, label=label)
                            dataset.add_record(record)
                        #print ""
                        #self.print_("Error: Input nodes doesn't match amount of pixels in: {}".format(fn), c="red",c_attrs=["bold"])
                        #exit(1)

                        #pass
                        #print "{} could not be added due to pixel size".format(fn)
                    percent = str((int(float(j+1)/len(files)*100)))

                    sys.stdout.flush()
                    self.print_("\r\t{}% ({}/{})".format(percent, i+1, len(args.paths)), use_prog=False, trailing=True)

        elif args.type == "csv":
            self.print_("Inputting CSV into dataset...", c="magenta", c_attrs=["bold"], trailing=True)
            sys.stdout.flush()

            for path in args.paths:
                for i, fn in enumerate(glob.glob(path)):
                    dataset.input_csv(fn, training="training"==args.set)
            self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)

        if os.path.exists(tmp_dir):
            rmtree(tmp_dir)

        self.print_("Saving Dataset ({}.dataset.pkl)...".format(args.out), c="magenta", c_attrs=["bold"], trailing=True)
        sys.stdout.flush()
        dataset.save(args.out)
        self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)



    def do_test(self, command):
        test_parser = argparse.ArgumentParser(prog=self.prog)
        test_parser.add_argument('-d', '--data', required=True, dest="fn")
        test_parser.add_argument('-i', '--images', default=None, nargs="*")
        test_parser.add_argument('-p', '--paths', default=None, nargs="*")
        args = test_parser.parse_args(command)

        n = NeuralNetwork.load(args.fn)
        results = []
        records = []

        if args.images is None and args.paths is None:
            self.print_("error: At least arg --images or arg --path has to be set", c="red", c_attrs=["bold"])
            exit(1)

        if args.images is not None:
            images = self.pair_args(args.images)

            self.print_("Starting image analysis...", c="magenta", c_attrs=["bold"], trailing=True)
            for image_info in images:
                record = Record()
                record.input_image(image_info[0], label=image_info[1])
                records.append(record)
            self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)


        if args.paths is not None:
            for i, path in enumerate(args.paths):
                d = None
                d = Dataset.load(path)
                if d is not None:
                    records +=d.records

        for i, record in enumerate(records):
            percent = str((float(i+1)/len(records))*100)[:5]

            sys.stdout.flush()
            self.print_('\r\t{}%'.format(percent), trailing=True, use_prog=False)

            try:
                response = n.query(record)
            except InputNodesException:
                self.print_("ERROR: Input nodes are not equal to the pixels", c="red", c_attrs=["bold"], use_prog=False)
            else:
                results.append({"label":record.label, "response":response.label, "elapsed":response.elapsed_ms})
        self.print_(' OK', c="blue", c_attrs=["bold"], use_prog=False)

        print("")

        if len(results) !=0:
            self.print_("\tRESULTS", c="magenta", c_attrs=["bold"], use_prog=False)
            num_correct = 0
            elapsed_time = 0
            for result in results:
                match = int(result["label"]) == result["response"]
                self.print_("\t  Label: {}".format(result["label"]), use_prog=False)
                self.print_("\t  Response: {}".format(result["response"]), use_prog=False)
                self.print_("\t  Match:", use_prog=False, trailing=True)

                elapsed_time+=result["elapsed"]
                if match:
                    self.print_("TRUE", c="green", c_attrs=["bold"], use_prog=False)
                    num_correct+=1
                else:
                    self.print_("FALSE", c="red", c_attrs=["bold"], use_prog=False)

                self.print_("\t  Elapsed (ms): {}".format(result["elapsed"]), use_prog=False)

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
        train_parser = argparse.ArgumentParser(prog=self.prog+' train')

        train_parser.add_argument('-d', '--data', type=str, required=True, dest="datasets", nargs="*",
                                        help="path to the dataset file")
        train_parser.add_argument('-i', '--inodes', type=int,
                                        help="Number of input nodes")
        train_parser.add_argument('-o', '--onodes', type=int,
                                        help="Number of output nodes")
        train_parser.add_argument('-n', '--hnodes', type=int, required=True,
                                        help="Number of hidden nodes")
        train_parser.add_argument('-l', '--l_rate', type=float, default=0.1,
                                        help="Learning rate")
        train_parser.add_argument('-e', '--epoch', type=int, default=5,
                                        help="Numbers of iteration data set will be used in training")
        train_parser.add_argument('--out', type=str, default="out")

        args = train_parser.parse_args(command)

        self.print_("Creating Network object..", c="magenta", c_attrs=["bold"], trailing=True)
        sys.stdout.flush()
        temp_ds = Dataset.load(args.datasets[0])

        n = NeuralNetwork(temp_ds.input_nodes, args.hnodes, temp_ds.output_nodes, args.l_rate)
        self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)

        #self.print_("Converting Images to arrays...", c="magenta", c_attrs=["bold"])
        #data = []
        #dataset = Dataset(n.input_nodes, n.output_nodes)
        #for i, csv in enumerate(args.fn):
        #    dataset.input_csv(csv, training=True)
        #    sys.stdout.flush()
        #    self.print_("\r\t({}/{})".format(i+1, len(args.fn)), trailing=True, use_prog=False)
        #self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)



        self.print_("Training Network...", c="magenta", c_attrs=["bold"])
        errors = 0
        for dataset_fn in args.datasets:
            dataset = Dataset.load(dataset_fn)
            for j, record in enumerate(dataset.records):
                #for i in xrange(0, len(record.inputs)):

                percent = str((float(j+1)/len(dataset.records))*100)[:5]
                data = n.train(record, args.epoch)

                sys.stdout.flush()
                self.print_("\r\t{}% Error: {}".format(percent, data.output_errors[len(data.output_errors)-1][0]),  c_attrs=["bold"], use_prog=False, trailing=True)

            self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)


        self.print_("Saving Network ({}.nn.pkl)...".format(args.out), c="magenta", c_attrs=["bold"], trailing=True)
        n.save(args.out)
        self.print_(" OK", c="blue", c_attrs=["bold"], use_prog=False)
        self.print_("ERRORS: {}".format(errors))

    def do(self, command):
        if command[0] in CharClassifierCommand.do_map:
            getattr(self, CharClassifierCommand.do_map[command[0]])(command[1:])

        else:
            print CharClassifierCommand.usage
            print("{}: {} is an unknown command".format(self.prog, command[0]))


if __name__ == "__main__":

    if len(sys.argv)<2:
        print CharClassifierCommand.usage
        exit(1)
    
    ccc = CharClassifierCommand()
    ccc.do(sys.argv[1:])




