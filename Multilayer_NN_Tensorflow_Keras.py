# -*- coding: utf-8 -*-

#Import Dataset 

import pandas as pd

df_test = pd.read_csv("../Datasets/mnist_dataset/mnist_test.csv",
                 header=None)

df_train = pd.read_csv("../Datasets/mnist_dataset/mnist_train.csv",
                 header=None)

X_train = df_train.iloc[:,1:].values
y_train = df_train.iloc[:,:1].values.reshape(-1)
X_test = df_test.iloc[:,1:].values
y_test = df_test.iloc[:,:1].values.reshape(-1)

del df_train, df_test
#Data Preprocessing 

#Normalization : MinMaxScaler

from sklearn.preprocessing import MinMaxScaler 
MMS = MinMaxScaler()
X_train_norm = MMS.fit_transform(X_train)
X_test_norm = MMS.transform(X_test)

del X_train, X_test

"""  Implementation of Multilayer Neural Network with Keras """
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np 
# NOTE:
# ================================================
# If you have TensorFlow v1.3 installed
# you can use the keras API by
# importing it from the contrib module
# `import tensorflow.contrib.keras as keras`

np.random.seed(123)
tf.set_random_seed(123)

#datapreprocessing 
y_train_onehot = keras.utils.to_categorical(y_train)

#Model Building
model = keras.models.Sequential()

model.add(
    keras.layers.Dense(
        units=50,    
        input_dim=X_train_norm.shape[1],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=50,    
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='tanh'))

model.add(
    keras.layers.Dense(
        units=y_train_onehot.shape[1],    
        input_dim=50,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        activation='softmax'))


sgd_optimizer = keras.optimizers.SGD(
        lr=0.001, decay=1e-7, momentum=.9)

model.compile(optimizer=sgd_optimizer,
              loss='categorical_crossentropy')

#Training 

history = model.fit(X_train_norm, y_train_onehot,
                    batch_size=64, epochs=50,
                    verbose=1,
                    validation_split=0.1)


#Evaluation 
y_test_pred = model.predict_classes(X_test_norm, 
                                    verbose=0)

correct_preds = np.sum(y_test == y_test_pred, axis=0) 
test_acc = correct_preds / y_test.shape[0]
print('Test accuracy: %.2f%%' % (test_acc * 100))

#Generalization on a single Sample 
print (np.argmax(model.predict(X_test_norm[0].reshape((1,-1)))))


