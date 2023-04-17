import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import json
import sys
import time
import os
from os import path
import gc


# add the relative path of the parent directory
sys.path.insert(0, '../')
gc.collect()

# The command for excuting this script should be:
# python3 model.py $dataFileName $jsonName
# $dataFileName: the name of sequence/label set with out the prefix and extension
# $jsonName: full json file name with extension
# example: python3 model.py fake01 mlp.json

# get the name of this script
scriptName = sys.argv[0].split('/')[-1]

# get the name of our data/label file
dataName = sys.argv[1]
seqFile = "sequence-" + dataName + ".in"
labelFile = "label-" + dataName + ".in"

# Now we get the name of json file
jsonName = sys.argv[2]
# print out all our arguments to make sure we are executing correctly
print("==================================================================================")
print("Executing "+scriptName+" on data "+seqFile+" and model spec "+jsonName, flush=True)
print("==================================================================================")

# Load the json file
jsonFile = open("../model-spec/" + jsonName)
jsonData = json.load(jsonFile)

# get the model spec
# FIXME
# load all possible model spec from json file regardless of what model we are runnning
# if the model parameter is not available for this model ("convolution size for RNN"), leave the entry in JSON file empty.
model = jsonData["modelName"]       # The model we want to run in this script (CNN, RNN...)
dataRoot = jsonData["dataRoot"]     # The path root of the dataset (../dataset/)
outRoot = jsonData["outRoot"]       # The path root where we want to save the output
ngpu = jsonData["ngpu"]             # The number of gpu we are going to use
lr = jsonData["lr"]                 # The learning rate of our current model
batchSize = jsonData["batchSize"]   # The batch size of the current training procedure
nEpoch = jsonData["nEpoch"]         # Maximum number of epoches we want to run (might early stop)
lrSteps = jsonData["lrSteps"]       # The maximum number of time we want to decrease our learning rate

print("Finish loading from " + jsonName, flush=True)
print("We are running the {} model".format(model))
print("The learning rate is {}".format(lr))
print("The batch size is {}".format(batchSize))
print("The number of Epoch is {}".format(nEpoch))
print("==================================================================================")

# now we load the input sequences and labels as lists of strings
with open(dataRoot + seqFile, 'r') as f:
    seqString = f.readlines()
with open(dataRoot + labelFile, 'r') as f:
    labelChar = f.readlines()

# get the sample size and sequence length
nSamples = len(labelChar)
lenSequence = len(seqString[0]) - 1

print("Finish loading data from " + seqFile + " and " + labelFile, flush=True)
print("The number of sample is {}".format(nSamples))
print("The length of each sequence is {}".format(lenSequence))
print("==================================================================================")

# TODO: encode the sequence
# TODO: define the models. We want all models in this script, and we only call the specifc model in the training defined by {$model} in json
# TODO: training and testing process
# TODO: performance analysis and saving the result