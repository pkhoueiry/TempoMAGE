#!/usr/bin/python
'''
Training tempoMAGE and generating performance plots

'''

import os
import glob
import sys, getopt
import tempfile
from datetime import datetime
#from packaging import version

if len(sys.argv)==1:
        print('train_tempoMAGE.py','\n'
        '     -t/--training_data <full path for the training dataset FILES>\n',
        '    -o/--output_dir <output directory for plots>\n', 
        '    -e/--epochs <EPOCHS>\n',
        '    -s/--save <whether or not to save the trained model>\n')
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
from load_data import plot_prc, plot_roc, plot_cm, prepare_training_data

METRICS = set_metrics()
default_EPOCHS = 50
BATCH_SIZE = 500
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_crossentropy', 
    verbose=1,
    patience=5,
    restore_best_weights=True)


opts, args = getopt.getopt(sys.argv[1:],'ht:o:e:s:',['help','training_data=','output_dir=','epochs=','save='])

datapath=""
plotpath=""
EPOCHS=""
save_model=""


for opt, arg in opts:        
    if opt in ('-h', '--help') :
        print('train_tempoMAGE.py',
        '-t/--training_data <full path for the training dataset FILES>',
        '-o/--output_dir <output directory for plots and trained model>', 
        '-e/--epochs <EPOCHS>',
        '-s/--save <whether or not to save the trained model')
        sys.exit()
    
    elif opt in ('-t', '--training_data'):
        datapath = arg
    
    elif opt in ('-o', '--output_dir'):
        plotpath = arg

    elif opt in ('-e', '--epochs'):
        EPOCHS = int(arg)
    
    elif opt in ('-s', '--save'):
        save_model = arg

if EPOCHS == "":
    print("Training with default number of epochs = {}".format(default_EPOCHS))
    EPOCHS = default_EPOCHS
else:
    print("Training with user-defined number of epochs = {}".format(EPOCHS))
    EPOCHS = EPOCHS

# Load the training dataset
print("Preparing training and validation data... \n")
(depth_train, depth_val, exp_train, exp_val, weight_train,
 weight_val, seq_train, seq_val, y_train, y_val,train_bed, val_bed)= prepare_training_data(datapath)


tempoMAGE = tempoMAGE(metrics=METRICS)
print("Starting tempoMAGE training at: " + datetime.now().strftime("%Y%m%d_%H:%M"))
history = tempoMAGE.fit([seq_train, depth_train,exp_train, weight_train],[y_train],
                         epochs= EPOCHS, batch_size= BATCH_SIZE,
                         validation_data=([seq_val, depth_val,exp_val, weight_val],[y_val]),
                         verbose=1, callbacks=[early_stopping])
print("Finished tempoMAGE training at: " + datetime.now().strftime("%Y%m%d_%H:%M"))

# get model predictions
print("Running predictions on validation data:")
validation_results = tempoMAGE.evaluate([seq_val, depth_val, exp_val, weight_val], y_val,
                                  batch_size=BATCH_SIZE, verbose=1)
validation_results = np.around(validation_results, decimals=3)
for name, value in zip(tempoMAGE.metrics_names,validation_results):
  print(name, ': ', value)
print()
y_pred = tempoMAGE.predict([seq_val, depth_val, exp_val, weight_val], batch_size=BATCH_SIZE)
y_pred = np.around(y_pred, decimals=2)
# assign prediction score of validation data to the output bed file
val_bed['Score'] = y_pred
val_bed.to_csv(os.path.join(plotpath,"validation_prediction.bed"), sep="\t",index=False,header=False )

# collect performance metrics 
precision, recall, _ = sklearn.metrics.precision_recall_curve(y_val, y_pred)
f1_score = 2*precision*recall/(precision+recall)
print("The f1_score is :" + str(f1_score))

# plot auPRC 
name = "auPRC = " + str(validation_results[9])
plot_prc(name, y_val,y_pred, color=colors[0], linestyle = "-")
plt.plot(f1_score*100, label = "F1_score", color = colors[4], linestyle=":")
plt.legend(loc='lower left')
plt.savefig(os.path.join(plotpath,"auPRC.pdf"))
plt.close()

# plot auROC
name = "auROC = " + str(validation_results[8])
plot_roc(name, y_val,y_pred, color=colors[0], linestyle = "-")
plt.legend(loc='lower right')
plt.savefig(os.path.join(plotpath,"auROC.pdf"))
plt.close()

# plot confusion matrix
plot_cm(y_val, y_pred)
plt.savefig(os.path.join(plotpath,"confusion_matrix.pdf"))

print("The Score column in file \"validation_prediction.bed\" contains the predicted class for each tile in the validation dataset.\n")
print("Score > 0.5 = predicted positive Tile;Score < 0.5 = predicted negative Tile ")
if save_model in('yes','Yes','YES','y','Y'):
  tempoMAGE.save(os.path.join(plotpath,'tempoMAGE_savedmodel'))

print("Done!")

