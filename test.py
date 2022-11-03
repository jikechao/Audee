import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, \
    GlobalAveragePooling2D, ZeroPadding2D, Flatten, Dense, Activation, ReLU, LeakyReLU, PReLU, ELU, ThresholdedReLU, \
    BatchNormalization, SimpleRNN, LSTM, GRU, Embedding, Dropout, GaussianNoise, ConvLSTM2D, Conv2DTranspose, Input
import numpy as np
from myLayers import get_layer


input_layer = Input((288, 288, 3))
layer0 = get_layer("Conv2D")(input_layer)
layer = get_layer("AveragePooling2D")(layer0)
layer = get_layer("Conv2D")(layer)
layer2 = get_layer("AveragePooling2D")(layer)


model = keras.Model(inputs=input_layer, outputs=layer2)
model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
# model.save('demo_2.h5')
#
# model = keras.models.load_model('demo_2.h5')
model.summary()
