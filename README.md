# charc

####Command-line interface to easily train and test character classification ML.

Currently Date has to be in a csv file and the total number of pixels have to be the same for training and testing images


##Test
###Usage: Loads pickle of previously trained neuron network to use with testing images

####Command:
```
charc test [-h] -d pkl-filename  -i [IMAGES [IMAGES ...]]
```

####Args:
- -d: Pickle filename
- -i: name of image followed by label

####Examples: 
```
charc test -d nn.pkl -i image_number_1.jpg 1 image_number4.jpg 4

charc test -d other_nn.pkl -i number_5_image.png 5

```
##Train
###Usage: Trains and saves neuron network to pickle

####Command: 
```
charc train [-h] -d [training-data-filename[DATA ...] [-i INODES] [-o ONODES] [-n HNODES] [-l L_RATE]
             [-e EPOCH] [--out OUT]
```


####Args:
- -d: CSV files
- -i: Input Nodes
- -o: Output Nodes
- -n: Hidden Nodes
- -l: Learning Rate
- -e: epoch
- --out: Output filename

####Examples: 
```
charc test -d data.csv -e 3 -l .1 --out nn

charc test -d data.csv data2.csv -i 784 -n 200 -o 10 --out other_nn

```
