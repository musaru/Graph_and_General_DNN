# -*- coding: utf-8 -*-
"""
Created on Tue May 24 23:01:31 2022

@author: shinlab
"""

force_use_cpu = False

if force_use_cpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    from tensorflow.keras import backend as K
    print('There should not be a GPU in the following list:', K.tensorflow_backend._get_available_gpus())

import numpy
import tensorflow
import tensorflow.keras as keras
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D, Dropout, Flatten, Lambda, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

# model
dropout_probability = 0.2
duration = 100
n_classes = 14
n_channels = 66  # usually  n_channels = 2 * n_joints  or  n_channels = 3 * n_joints


def create_model(n_classes, duration, n_channels, dropout_probability=0.2):
    # Define model, using functional API
    model_input = Input(shape=(duration, n_channels))

    # slice into channels
    channel_inputs = Lambda(lambda x: tensorflow.split(x, num_or_size_splits=n_channels, axis=-1))(model_input)

    features = []
    for channel in range(n_channels):
        channel_input = channel_inputs[channel]
        # high branch
        high = Conv1D(filters=8, kernel_size=7, padding='same', activation='relu', input_shape=(100, 66))(channel_input)
        high = AveragePooling1D(pool_size=2)(high)
        high = Conv1D(filters=4, kernel_size=7, padding='same', activation='relu')(high)
        high = AveragePooling1D(pool_size=2)(high)
        high = Conv1D(filters=4, kernel_size=7, padding='same', activation='relu')(high)
        high = Dropout(dropout_probability)(high)
        high = AveragePooling1D(pool_size=2)(high)
        # low branch
        low = Conv1D(filters=8, kernel_size=3, padding='same', activation='relu', input_shape=(100, 66))(channel_input)
        low = AveragePooling1D(pool_size=2)(low)
        low = Conv1D(filters=4, kernel_size=3, padding='same', activation='relu')(low)
        low = AveragePooling1D(pool_size=2)(low)
        low = Conv1D(filters=4, kernel_size=3, padding='same', activation='relu')(low)
        low = Dropout(dropout_probability)(low)
        low = AveragePooling1D(pool_size=2)(low)
        # pooling branch
        ap_residual = AveragePooling1D(pool_size=2, input_shape=(100, 66))(channel_input)
        ap_residual = AveragePooling1D(pool_size=2)(ap_residual)
        ap_residual = AveragePooling1D(pool_size=2)(ap_residual)
        # channel output
        channel_output = concatenate([high, low, ap_residual])
        features.append(channel_output)

    features = concatenate(features)
    features = Flatten()(features)
    features = Dense(units=1936, activation='relu')(features)

    model_output = Dense(units=n_classes, activation='softmax')(features)

    model = Model(inputs=[model_input], outputs=[model_output])
    return model

model = create_model(n_classes=n_classes, duration=duration, n_channels=n_channels, dropout_probability=dropout_probability)
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)


import pickle
# We load a gesture dataset:
#
#   x.shape should be (dataset_size, duration, channel)
#   y.shape should be (dataset_size, 1)


# If you want to use the DHG dataset, go to: https://colab.research.google.com/drive/1ggYG1XRpJ50gVgJqT_uoI257bspNogHj
use_dhg_dataset = True

if use_dhg_dataset:

    #try:
        # Connect Google Colab instance to Google Drive
        #from google.colab import drive
        #drive.mount('/gdrive')
        # Load the dataset (you already have created in the other notebook) from Google Drive
        #!cp /gdrive/My\ Drive/dhg_data.pckl dhg_data.pckl
    #except:
    #    print("You're not in a Google Colab!")

    def load_data(filepath='D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/dhg_data.pckl'):
        """
        Returns hand gesture sequences (X) and their associated labels (Y).
        Each sequence has two different labels.
        The first label  Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
        The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
        """
        file = open(filepath, 'rb')
        data = pickle.load(file, encoding='latin1')  # <<---- change to 'latin1' to 'utf8' if the data does not load
        file.close()
        return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data['y_test_28']

    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = load_data('D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/dhg_data.pckl')
    y_train_14, y_test_14 = numpy.array(y_train_14), numpy.array(y_test_14)
    y_train_28, y_test_28 = numpy.array(y_train_28), numpy.array(y_test_28)
    if n_classes == 14:
        y_train = y_train_14
        y_test = y_test_14
    elif n_classes == 28:
        y_train = y_train_28
        y_test = y_test_28

else:
    # Let's create fake data, with shape: (dataset_size, duration, channel)
    x_train = numpy.random.randn(2000, duration, n_channels)
    y_train = numpy.random.random_integers(n_classes, size=2000)

    x_test = numpy.random.randn(1000, duration, n_channels)
    y_test = numpy.random.random_integers(n_classes, size=1000)
    
import numpy as np
print("Xtrain=",np.array(x_train).shape)
print("Xtest=",np.array(x_test).shape)
print("Ytrain=",np.array(y_train_14).shape)
print("Ytrain=",np.array(y_train_28).shape)
print("Ytrain=",np.array(y_test_14).shape)
print("Ytrain=",np.array(y_test_28).shape)

print(n_classes)
from keras.utils import np_utils
y_trainenc=[]
y_testenc=[]
print(y_train.shape)
print(y_test)
print("Shape before one-hot encoding: ", y_train.shape, y_test.shape)
y_trainenc = np_utils.to_categorical(y_train, n_classes)
y_testenc = np_utils.to_categorical(y_test, n_classes)
#from tensorflow.keras.utils import to_categorical

#y_trainenc= to_categorical(y_train, dtype ="uint8")
#y_testenc= to_categorical(y_test, dtype ="uint8")
print("Shape after one-hot encoding: for y train", y_trainenc.shape)
print("Shape after one-hot encoding: for y test", y_testenc.shape)


#Model Training
# Training: Optimizer's Learning Rate
learning_rate = 0.001

# We use Adam to optimize a multi-class classification task
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
loss = 'categorical_crossentropy'
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Start the training

# ...with an existing validation set:
history = model.fit(x_train, y_trainenc, validation_data=(x_test, y_testenc), epochs=100, batch_size=32)

# ...or, if there is no validation set:
# history = model.fit(x_train, y_train, validation_split=0.33, epochs=100, batch_size=32)
# plot everything
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = load_data('D:/PhD/Skelton_Depth_VedioSign Language/Depth_Skelton/Pickle_file/dhg_data.pckl')