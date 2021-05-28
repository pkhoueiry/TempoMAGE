#!/usr/bin/python
'''
Testing tempoMAGE from saved tempMAGE model
 and generating performance plots

'''

import os
import glob
import sys, getopt
import tempfile
from datetime import datetime

if len(sys.argv)==1:
        print('test_tempoMAGE.py','\n'
        '     -t/--testing_data <full path for the testing dataset FILES>\n',
        '    -o/--output_dir <output directory for plots>\n', 
        '    -m/--model <full path for the model already trained>\n')
        sys.exit()

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (5, 5)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sklearn
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf 
from tensorflow import keras
import tempoMAGE
from tempoMAGE import tempoMAGE,set_metrics
import load_data
from load_data import plot_prc, plot_roc, plot_cm, prepare_test_data


METRICS = set_metrics()
EPOCHS = 50
BATCH_SIZE = 500


opts, args = getopt.getopt(sys.argv[1:],'ht:o:m:',['help','testing_data=','output_dir=','model=',])

datapath = ""
plotpath = ""
modelpath = ""

for opt, arg in opts:        
    if opt in ('-h', '--help') :
        print('test_tempoMAGE.py','\n'
        '     -t/--testing_data <full path for the testing dataset FILES>\n',
        '    -o/--output_dir <output directory for plots>\n', 
        '    -m/--model <full path for the model already trained>\n')
        sys.exit()
    
    elif opt in ('-t', '--testing_data'):
        datapath = arg
    
    elif opt in ('-o', '--output_dir'):
        plotpath = arg

    elif opt in ('-m', '--model'):
        modelpath = arg


# Load the test dataset
print("Preparing test dataset... \n")
(depth_test, exp_test, weight_test,seq_test,y_test,test_bed)= prepare_test_data(datapath)

#Load tempoMAGE from saved model
print("Loading tempoMAGE saved model \n")
tempoMAGE= tf.keras.models.load_model(os.path.join(modelpath,'tempoMAGE_savedmodel'), compile=False)
tempoMAGE.compile(loss='binary_crossentropy',
                 optimizer=keras.optimizers.Adam(0.001),
                 metrics= METRICS)


# get model predictions
print("Running tempoMAGE predictions on test dataset \n")
test_results = tempoMAGE.evaluate([seq_test, depth_test, exp_test, weight_test], y_test,
                                  batch_size=BATCH_SIZE, verbose=1)
test_results = np.around(test_results, decimals=3)
for name, value in zip(tempoMAGE.metrics_names,test_results):
  print(name, ': ', value)
print()
y_pred = tempoMAGE.predict([seq_test, depth_test, exp_test, weight_test], batch_size=BATCH_SIZE)
y_pred = np.around(y_pred, decimals=2)
# assign prediction score of validation data to the output bed file
test_bed['Score'] = y_pred
test_bed.to_csv(os.path.join(plotpath,"test_prediction.bed"), sep="\t",index=False,header=False )

# collect performance metrics 
precision, recall, _ = sklearn.metrics.precision_recall_curve(y_test, y_pred)
f1_score = 2*precision*recall/(precision+recall)
print("The f1_score is :" + str(f1_score))

# plot auPRC 
name = "auROC = " + str(test_results[9])
plot_prc(name, y_test,y_pred, color=colors[1], linestyle = "-")
plt.plot(f1_score*100, label = "F1_score", color = colors[4], linestyle=":")
plt.legend(loc='lower left')
plt.savefig(os.path.join(plotpath,"testdata_auPRC.pdf"))
plt.close()

# plot auROC
name = "auROC = " + str(test_results[8])
plot_roc(name, y_test,y_pred, color=colors[0], linestyle = "-")
plt.legend(loc='lower right')
plt.savefig(os.path.join(plotpath,"testdata_auROC.pdf"))
plt.close()

# plot confusion matrix
plot_cm(y_test, y_pred)
plt.savefig(os.path.join(plotpath,"testdata_confusion_matrix.pdf"))

print("The Score column in file \"test_prediction.bed\" contains the predicted class for each tile in the test dataset.")
print("Score > 0.5 = predicted positive Tile;Score < 0.5 = predicted negative Tile ")
print("Done!")

