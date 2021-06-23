#!/usr/bin/python
'''
Data loading and pre-processing functions

'''

import os
import glob
import tempfile
import numpy as np
import pandas as pd
import re
from Bio import SeqIO
import pyranges as pr
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import mlxtend
from mlxtend.plotting import plot_confusion_matrix



def string_to_array(my_string):
    """ Change a DNA string (actg or z) to an array of digits"""
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array 

def one_hot_encoder(my_array):
    """ One-hot encoding for sequence input data"""
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['z','a','c','g','t']))
    integer_encoded = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories=[([0,1,2,3,4])])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)   
    return onehot_encoded

def load_sequence_data(inputPath, sequenceFile):
    """ Load, encode, and parse sequence and label data """
    sequences= pd.read_csv(os.path.join(inputPath,sequenceFile))
    # examine class imbalance
    neg, pos = np.bincount(sequences['label'])
    total = neg + pos
    print('dataset Info:\n Total samples: {}\n Positive Tiles: {}({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))
    onehot = np.empty([sequences.shape[0],len(sequences['seq'][1]),5]) #dimension=[no.sequences,len.sequence,onehotencoding]
    for row in range(sequences.shape[0]):
        onehot[row] = one_hot_encoder(string_to_array(sequences.seq[row]))
    seq_data= onehot.reshape(onehot.shape[0],len(sequences['seq'][1]),5,1)
    seq_labels = sequences['label']
    return seq_data, seq_labels

def prepare_training_data(datapath,validation_split=0.1):
    """ Wrapper function to load the training dataset """
    
    print("Loading and encoding the read-depth data")
    depth_train = np.array(pd.read_csv(os.path.join(datapath,'train_depth.txt'),sep="\t", header = None))
    depth_train = depth_train.reshape(depth_train.shape[0],depth_train.shape[1], 1)
    print("Finished loading and encoding the read-depth data")

    print("Loading and encoding the gene-expression data")
    exp_train = np.array(pd.read_csv(os.path.join(datapath,'train_expression.txt'),sep="\t", header = None))
    exp_train = exp_train.reshape(exp_train.shape[0], exp_train.shape[1],1)
    print("Finished loading and encoding the gene-expression data")

    print("Loading and encoding the reference time-point data")
    time_train = np.array(pd.read_csv(os.path.join(datapath,'train_ref.txt'),sep="\t", header = None))
    time_train = time_train.reshape(time_train.shape[0], time_train.shape[1], 1)
    print("Finished loading and encoding the reference time-point data")

    print("Loading foldchange data")
    foldchange_train = np.array(pd.read_csv(os.path.join(datapath,'train_foldchange.txt'),sep="\t", header = None))
    foldchange_train = foldchange_train.reshape(foldchange_train.shape[0], foldchange_train.shape[1], 1)
    print("Finished loading and encoding the foldchange data")

    print("Loading and one-hot encoding the sequence data")
    weight_train = time_train*foldchange_train
    seq_train, y_train = load_sequence_data(datapath, 'train_sequences.csv')
    print("The number of positive and negative tiles in the train dataset is:")
    print(y_train.value_counts())
    
    train_bed= pr.read_bed(os.path.join(datapath,"train_tiles.bed"),
                                as_df=True)

    print('Splitting data into: {}% training and {}% validation\n'.format(
        (1- validation_split)*100, validation_split *100))
    
    (depth_train,depth_val,seq_train,seq_val,exp_train,exp_val,y_train,y_val,weight_train,
    weight_val, train_bed, val_bed) = train_test_split(depth_train,seq_train,exp_train,y_train,weight_train,train_bed,
    test_size = validation_split, random_state = 50)

    print('Training labels shape:', y_train.shape)
    print('Validation labels shape:', y_val.shape)
    print('Training features shape:', depth_train.shape, seq_train.shape, exp_train.shape, weight_train.shape)
    print('Validation features shape:', depth_val.shape, seq_val.shape, exp_val.shape, weight_val.shape)

    return depth_train, depth_val, exp_train, exp_val, weight_train, weight_val, seq_train, seq_val, y_train, y_val,train_bed, val_bed

def prepare_test_data(datapath):
    """ Wrapper function to load the test dataset """

    print("Loading and encoding the test dataset")
    depth_test = np.array(pd.read_csv(os.path.join(datapath,'test_depth.txt'),sep="\t", header = None))
    depth_test = depth_test.reshape(depth_test.shape[0],depth_test.shape[1], 1)
    exp_test = np.array(pd.read_csv(os.path.join(datapath,'test_expression.txt'),sep="\t", header = None))
    exp_test = exp_test.reshape(exp_test.shape[0], exp_test.shape[1],1)
    time_test = np.array(pd.read_csv(os.path.join(datapath,'test_ref.txt'),sep="\t", header = None))
    time_test = time_test.reshape(time_test.shape[0], time_test.shape[1], 1)
    foldchange_test = np.array(pd.read_csv(os.path.join(datapath,'test_foldchange.txt'),sep="\t", header = None))
    foldchange_test = foldchange_test.reshape(foldchange_test.shape[0], foldchange_test.shape[1], 1)
    weight_test = time_test*foldchange_test
    seq_test, y_test = load_sequence_data(datapath, 'test_sequences.csv')
    test_bed= pr.read_bed(os.path.join(datapath,"test_tiles.bed"),
                                as_df=True)

    print('Test labels shape:', y_test.shape)
    print('Test features shape:', depth_test.shape, seq_test.shape, exp_test.shape, weight_test.shape)

    return depth_test, exp_test, weight_test, seq_test, y_test, test_bed

def plot_roc(name, labels, predictions, **kwargs):
    """ auROC plotting function """ 
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(fp, tp, label=name, linewidth=1, **kwargs)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.005,1.02])
    plt.ylim([-0.005,1.02])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_prc(name, labels, predictions, **kwargs):
    """ auPRC plotting function """
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.plot(100*recall, 100*precision, label=name, linewidth=1, **kwargs)
    plt.xlabel('Recall [%]')
    plt.ylabel('Precision [%]')
    plt.xlim([-0.5,102])
    plt.ylim([-0.5,102])
    plt.legend(loc='lower left')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_cm(labels, predictions, p=0.5):
    """ confusion matrix plotting function """
    cm = confusion_matrix(labels, predictions > p)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    figsize=(8,8),
                                    class_names=['False','True'],
                                    show_normed=True) 
    plt.tight_layout()
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print('Negative Tiles Correctly Classified (True Negatives): ', cm[0][0])
    print('Negative Tiles Incorrectly Classified (False Positives): ', cm[0][1])
    print('Positive Tiles Correctly Classified (True Positives): ', cm[1][1])
    print('Positive Tiles Incorrectly Classified (False Negatives): ', cm[1][0])
    print('Total Positive Tiles: ', np.sum(cm[1]))

