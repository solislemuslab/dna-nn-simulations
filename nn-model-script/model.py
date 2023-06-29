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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from itertools import product

from transformers import AutoTokenizer

import models

# add the relative path of the parent directory
sys.path.insert(0, '../')
gc.collect()

# prompt for usage
if len(sys.argv) != 3:
    print("Usage: python3 model.py $dataFileName $jsonName")
    print("$dataFileName: the name of sequence/label set without the prefix and extension")
    print("$jsonName: full json file name with extension")
    print("example: python3 model.py fake01 mlp.json")
    exit(0)

# The command for executing this script should be:
# python3 model.py $dataFileName $jsonName
# $dataFileName: the name of sequence/label set without the prefix and extension
# $jsonName: full json file name with extension
# example: python3 model.py fake01 mlp.json

# get the name of this script
scriptName = sys.argv[0].split('/')[-1]

# get the name of our data/label file
dataName = sys.argv[1]
seqFile = "sequence-" + dataName + ".in"
labelFile = "label-" + dataName + ".in"

# get the name of json file
jsonName = sys.argv[2]

# print out all our arguments to make sure we are executing correctly
print("==================================================================================")
print("Executing " + scriptName + " on data " + seqFile + " and model spec " + jsonName, flush=True)
print("==================================================================================")

# load the json file
jsonFile = open("../model-spec/" + jsonName)
jsonData = json.load(jsonFile)

# get the model spec
# FIXME
# load all possible model spec from json file regardless of what model we are running
# If the model parameter is not available for this model ("convolution size for RNN"),
# leave the entry in JSON file empty.
modelName = jsonData["modelName"]       # The model we want to run in this script (CNN, RNN...)
dataRoot = jsonData["dataRoot"]     # The path root of the dataset (../dataset/)
outRoot = jsonData["outRoot"]       # The path root where we want to save the output
nGpu = jsonData["nGpu"]             # The number of gpu we are going to use
lr = jsonData["lr"]                 # The learning rate of our current model
batchSize = jsonData["batchSize"]   # The batch size of the current training procedure
nEpoch = jsonData["nEpoch"]         # Maximum number of epochs we want to run (might early stop)
lrSteps = jsonData["lrSteps"]       # The maximum number of time we want to decrease our learning rate

print("Finish loading from " + jsonName, flush=True)
print("We are running the {} model".format(modelName))
print("The learning rate is {}".format(lr))
print("The batch size is {}".format(batchSize))
print("The number of Epoch is {}".format(nEpoch))
print("==================================================================================")

# load the input sequences and labels as lists of strings
with open(dataRoot + seqFile, 'r') as f:
    sequences = f.readlines()
    sequences = [sequence.strip() for sequence in sequences]
with open(dataRoot + labelFile, 'r') as f:
    labels = f.readlines()
    labels = [label.strip() for label in labels]

# get the sample size and sequence length
sample_size = len(labels)
seq_length = len(sequences[0])

print("Finish loading data from " + seqFile + " and " + labelFile, flush=True)
print("The number of sample is {}".format(sample_size))
print("The length of each sequence is {}".format(seq_length))
print("==================================================================================")


# kmer encoding
def seq2kmer(seq, k) -> str:
    """
    converts original sequences to kmers

    seq: str, original sequence
    k: int, the length of each kmer

    returns str: kmers separated by spaces
    """
    kmers = [seq[x: x + k] for x in range(len(seq) + 1 - k)]

    return " ".join(kmers)


# get the kmer encodings
K = 6
kmer_encodings = [seq2kmer(x, K) for x in sequences]

# for i in range(5):
#     print(kmer_encodings[i])

# After the kmer encoding process, the next step is either using one-hot encoding or tokenizer
# to "tokenize" the data (words/kmers) into numerical forms that are acceptable for models
# which can be done in the process of creating the datasets.

# # Tokenizer test
# # load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
#
# # tokenize the data
# tokenized_data = tokenizer(kmer_encodings, return_tensors="pt")
#
# print(tokenized_data['input_ids'])
# print(len(np.array(tokenized_data["input_ids"])))
# print(np.array(tokenized_data["input_ids"]))
# print(labels)
# print(len(np.array(labels).astype(int)))
# print(np.array(labels).astype(int))

# # One-hot encoding test
# # The LabelEncoder encodes a sequence of bases as a sequence of integers.
# integer_encoder = LabelEncoder()
# # The OneHotEncoder converts an array of integers to a sparse matrix where
# # each row corresponds to one possible value of each feature.
# one_hot_encoder = OneHotEncoder(categories='auto')
#
# one_hot_encodings = []
# for sequence in sequences:
#     integer_encoded = integer_encoder.fit_transform(list(sequence))
#     integer_encoded = np.array(integer_encoded).reshape(-1, 1)
#     one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
#     one_hot_encodings.append(one_hot_encoded.T.toarray())
#
# one_hot_encodings = np.stack(one_hot_encodings)
# # print(one_hot_encodings)
# # print(type(one_hot_encodings))
# # print(one_hot_encodings.shape)
#
# X = one_hot_encodings
# y = np.array(labels).astype(int)
# # split data into training data (90%) and testing data (10%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))
# X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.167, random_state=42)
# print(len(X_train))
# print(len(X_dev))
# print(len(y_train))
# print(len(y_dev))

# # kmer count test
# # set the value of k
# k = 3
# # get all the possible kmers
# dictionary = [p for p in product(["A", "C", "G", "T"], repeat=k)]
# # initialization: all zeros
# zeros = np.zeros((len(sequences), len(dictionary)))
#
# # iterate through all the sequences
# for i in range(0, len(sequences)):
#     # iterate each base-pair in a sequence
#     for j in range(0, seq_length - k + 1):
#         # counting for each kmer
#         for index, kmer in enumerate(dictionary):
#             if kmer == tuple(sequences[i][j:j + k]):
#                 zeros[i][index] += 1
#                 break
# print(dictionary)
# print(len(dictionary))
# print(zeros)
# print(zeros.shape)  # (len(sequences), len(dictionary))
# a = zeros.reshape((sample_size, 1, (4 ** k)))
# print(a)
# print(a.shape)
# print(type(zeros))  # numpy.ndarray
# print(zeros.dtype)  # float64


# Done: create the datasets for training, evaluating, and testing
# Here we have 3 different ways of embedding/encoding:
# 1. naive one-hot encoding; 2. kmer count; 3. Tokenizer


# input shape for the model
INPUT_SHAPE = ()


def dataset(mode='train', encoding='one-hot'):
    """
    Return dataset with specific encoding method
    Encoding/Embedding methods: one-hot, kmer-count, tokenizer
    """
    class NaiveOneHotDataset(data.Dataset):
        """Dataset created through naive one-hot encoding"""
        def __init__(self):
            self.mode = mode

            # The LabelEncoder encodes a sequence of bases as a sequence of integers.
            integer_encoder = LabelEncoder()
            # The OneHotEncoder converts an array of integers to a sparse matrix where
            # each row corresponds to one possible value of each feature.
            one_hot_encoder = OneHotEncoder(categories='auto')

            one_hot_encodings = []
            for sequence in sequences:
                integer_encoded = integer_encoder.fit_transform(list(sequence))
                integer_encoded = np.array(integer_encoded).reshape(-1, 1)
                one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
                one_hot_encodings.append(one_hot_encoded.T.toarray())

            one_hot_encodings = np.stack(one_hot_encodings)

            X = one_hot_encodings
            y = np.array(labels).astype(int)

            globals()['INPUT_SHAPE'] = X[0].shape

            # split data into training data (90%) and testing data (10%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            if mode == 'test':
                # testing data
                # convert data into PyTorch tensors
                self.data = torch.FloatTensor(X_test)
                self.target = torch.FloatTensor(y_test)
            else:
                # training data (train/validate)
                # split training data (90%) into train (75%) & dev (15%) sets
                X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.167, random_state=42)
                if mode == 'train':
                    self.data = torch.FloatTensor(X_train)
                    self.target = torch.FloatTensor(y_train)
                elif mode == 'dev':
                    self.data = torch.FloatTensor(X_dev)
                    self.target = torch.FloatTensor(y_dev)

            print('Finished reading the {} set of NaiveOneHotDataset ({} samples found)'.format(mode, len(self.data)))

        def __getitem__(self, index):
            # Returns one sample at a time
            # if self.mode in ['train', 'dev']:
            #     # for training
            #     return self.data[item], self.target[item]
            # else:
            #     # for testing (no target)
            #     return self.data[item]
            return self.data[index], self.target[index]

        def __len__(self):
            # Returns the size of the dataset
            return len(self.data)

    class KmerCountDataset(data.Dataset):
        """Dataset created by counting the kmers"""
        def __init__(self):
            self.mode = mode

            # set the value of k
            k = 6
            # get all the possible kmers
            dictionary = [p for p in product(["A", "C", "G", "T"], repeat=k)]
            # initialization: all zeros
            zeros = np.zeros((len(sequences), len(dictionary)))

            # iterate through all the sequences
            for i in range(0, len(sequences)):
                # iterate each base-pair in a sequence
                for j in range(0, seq_length - k + 1):
                    # counting for each kmer
                    for index, kmer in enumerate(dictionary):
                        if kmer == tuple(sequences[i][j:j + k]):
                            zeros[i][index] += 1
                            break

            X = zeros.astype(int).reshape((sample_size, 1, (4 ** k)))  # convert to 2d
            y = np.array(labels).astype(int)

            globals()['INPUT_SHAPE'] += X[0].shape

            # split data into training data (90%) and testing data (10%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            if mode == 'test':
                # testing data
                # convert data into PyTorch tensors
                self.data = torch.FloatTensor(X_test)
                self.target = torch.FloatTensor(y_test)
            else:
                # training data (train/validate)
                # split training data (90%) into train (75%) & dev (15%) sets
                X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.167, random_state=42)
                if mode == 'train':
                    self.data = torch.FloatTensor(X_train)
                    self.target = torch.FloatTensor(y_train)
                elif mode == 'dev':
                    self.data = torch.FloatTensor(X_dev)
                    self.target = torch.FloatTensor(y_dev)

            print('Finished reading the {} set of KmerCountDataset ({} samples found)'.format(mode, len(self.data)))

        def __getitem__(self, index):
            # Returns one sample at a time
            # if self.mode in ['train', 'dev']:
            #     # for training
            #     return self.data[item], self.target[item]
            # else:
            #     # for testing (no target)
            #     return self.data[item]
            return self.data[index], self.target[index]

        def __len__(self):
            # Returns the size of the dataset
            return len(self.data)

    class TokenDataset(data.Dataset):
        """Dataset created via tokenization on kmers"""
        def __init__(self):
            self.mode = mode

            # load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6", trust_remote_code=True)
            # tokenize the data (into tensors)
            tokenized_data = tokenizer(kmer_encodings, return_tensors="pt")
            # load data into numpy arrays
            X = np.array(tokenized_data["input_ids"])
            y = np.array(labels).astype(int)

            globals()['INPUT_SHAPE'] += X[0].shape

            # split data into training data (90%) and testing data (10%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            if mode == 'test':
                # testing data
                # convert data into PyTorch tensors
                self.data = torch.FloatTensor(X_test)
                self.target = torch.FloatTensor(y_test)
            else:
                # training data (train/validate)
                # split training data (90%) into train (75%) & dev (15%) sets
                X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.167, random_state=42)
                if mode == 'train':
                    self.data = torch.FloatTensor(X_train)
                    self.target = torch.FloatTensor(y_train)
                elif mode == 'dev':
                    self.data = torch.FloatTensor(X_dev)
                    self.target = torch.FloatTensor(y_dev)

            print('Finished reading the {} set of TokenDataset ({} samples found)'.format(mode, len(self.data)))

        def __getitem__(self, index):
            # Returns one sample at a time
            # if self.mode in ['train', 'dev']:
            #     # for training
            #     return self.data[item], self.target[item]
            # else:
            #     # for testing (no target)
            #     return self.data[item]
            return self.data[index], self.target[index]

        def __len__(self):
            # Returns the size of the dataset
            return len(self.data)

    # construct dataset based on encoding method specified
    if encoding == 'one-hot':
        onehot_dataset = NaiveOneHotDataset()
        return onehot_dataset
    elif encoding == 'kmer-count':
        kmer_count_dataset = KmerCountDataset()
        return kmer_count_dataset
    elif encoding == 'tokenizer':
        token_dataset = TokenDataset()
        return token_dataset


# def dataloader(mode, batch_size=batchSize, encoding='one-hot'):
#     """
#     Generate a dataset, then put it in a dataloader
#     """
#     # construct dataset
#     ds = dataset(mode=mode, encoding=encoding)
#     # construct dataloader
#     dl = data.DataLoader(ds, batch_size, shuffle=(mode == 'train'))  # only shuffle the training data
#     return dl

ENCODING = 'one-hot'
train_set = dataset('train', encoding=ENCODING)
train_loader = data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
val_set = dataset('dev', encoding=ENCODING)
val_loader = data.DataLoader(val_set, batch_size=batchSize, shuffle=False)
test_set = dataset('test', encoding=ENCODING)
test_loader = data.DataLoader(test_set, batch_size=batchSize, shuffle=False)

# Cleanup the unneeded variables to save memory.
del sequences, labels
gc.collect()

# Update: All the models will be defined in models.py for modularity
# We only call the specific model in the training defined by {$model} in json


# check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# get device
device = get_device()
print(f'DEVICE: {device}')


# TODO: training and testing process
def train():
    """
    The training process of DNN
    """
    # number of epochs
    n_epochs = nEpoch
    # learning rate
    learning_rate = lr
    # the path where checkpoint saved
    model_path = './model.ckpt'

    # create model, define a loss function, and optimizer
    model = getattr(models, modelName)(INPUT_SHAPE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    best_acc = 0.0
    for epoch in range(n_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            batch_loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        if len(val_set) > 0:
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    _, val_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability

                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                    val_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, n_epochs, train_acc / len(train_set), train_loss / len(train_loader),
                    val_acc / len(val_set), val_loss / len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, n_epochs, train_acc / len(train_set), train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')


train()


# TODO: performance analysis and saving the result
