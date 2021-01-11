#!/usr/bin/python
""" TempMAGE model architecture """


import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Conv1D,MaxPooling1D,Dense,Dropout

def set_metrics():
    """ metrics used to evaluate the model's perfromance """
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='ROC_auc'),
        keras.metrics.AUC(name='PR_auc', curve = "PR"),
        keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy')
    ]
    return METRICS


def tempoMAGE(metrics, output_bias= None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    seq_input = keras.Input(shape=(400,5,1), name='sequence_conv')
    x = keras.layers.Conv2D(filters=32, kernel_size=(10,5), padding='valid', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))(seq_input)
    x = keras.layers.MaxPooling2D(pool_size=(2,1))(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(2,1), padding='valid', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,1))(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(2,1), padding='valid', activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0005) )(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,1))(x)
    sequence_features = keras.layers.Flatten()(x)
 
    depth_input = keras.Input(shape=(400,1), name= 'depth')
    x = keras.layers.Conv1D(filters= 32, kernel_size=(5), padding='valid', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))(depth_input)
    x = keras.layers.MaxPooling1D(pool_size=(2))(x)
    x = keras.layers.Conv1D(filters= 64, kernel_size=(2), padding='valid', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = keras.layers.MaxPooling1D(pool_size=(2))(x)
    x = keras.layers.Conv1D(filters= 128, kernel_size=(2), padding='valid', activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = keras.layers.MaxPooling1D(pool_size=(2))(x)
    depth_features = keras.layers.Flatten()(x)
    
    x = layers.concatenate([sequence_features, depth_features])
    conv_dense = keras.layers.Dense(118, activation = 'relu')(x)
    
    expression_input = keras.Input(shape=(20,1), name= 'expression')
    expression_features = keras.layers.Flatten()(expression_input)
    
    time_input = keras.Input(shape=(1,1), name= 'time')
    time_features = keras.layers.Flatten()(time_input)
    
    x = layers.concatenate([expression_features, time_features])
    data_dense = keras.layers.Dense(10,activation = 'relu')(x)
    
    x = layers.concatenate([conv_dense, data_dense])
    x = keras.layers.Dense(128, activation = 'relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = keras.layers.Dropout(0.3)(x)
    seq_pred = keras.layers.Dense(1, activation='sigmoid',bias_initializer= output_bias)(x)
    model = keras.Model(inputs=[seq_input,depth_input,expression_input, time_input], outputs= seq_pred)
    model.compile(loss='binary_crossentropy',
                 optimizer=keras.optimizers.Adam(0.001),
                 metrics= metrics)
    return model
    