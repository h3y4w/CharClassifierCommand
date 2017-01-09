# charc

####Command-line interface to easily train and test character classification ML.

Currently Date has to be in a csv file and the total number of pixels have to be the same for training and testing images



Command:
```
charc test [-h] -d pkl-filename  -i [IMAGES [IMAGES ...]]
```
Usage: Loads pickle of previously trained neuron network to use with testing images
Example: 
```
charc test -d nn.pkl -i image_number_1.jpg:1
```

Command: 
```
charc train [-h] -d training-data-filename [-i INODES] [-o ONODES] [-n HNODES] [-l L_RATE]
             [-e EPOCH] [--out OUT]
```

Usage: Trains and saves neuron network to pickle

Args:
-i: Input Nodes
-o: Output Nodes
-n: Hidden Nodes
-l: Learning Rate
-e: epoch
-out: Output filename

Example: 
```
charc test -d data.csv -e 3 -l .1 --out nn 
```
