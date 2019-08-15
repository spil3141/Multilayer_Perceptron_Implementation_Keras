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


#predicting new sample

import PIL
import numpy as np 

def image_2_numpyArray(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    from PIL import Image, ImageFilter
    import numpy as np
    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    x=[tva]
    newArr=[[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
        for j in range(28):
            newArr[i][j]=x[0][k]
            k=k+1
    return np.asarray(newArr)



img = imageprepare("../Datasets/img.png")

img = np.asarray(newArr)


img = np.asarray(im)
data = img 
new_sample = np.asarray(data)
new_sample = MMS.transform(new_sample)

#Generalization on a single Sample 
print (np.argmax(model.predict(new_sample.reshape((1,-1)))))


