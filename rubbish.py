import numpy as np
np.random.seed(999)
import tensorflow as tf

import keras
from keras import layers

input1 = tf.keras.layers.Input(shape=(16,))
x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
input2 = tf.keras.layers.Input(shape=(32,))
x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
# equivalent to `added = tf.keras.layers.add([x1, x2])`
added = tf.keras.layers.Add()([x1, x2])
out = tf.keras.layers.Dense(4)(added)
model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
model.summary()

input_layers = {}
output_tensors = {}

for layer in model.layers:
    # print('............')
    # print(layer.name)
    for node in layer._outbound_nodes:
        layer_name = node.outbound_layer.name
        # print(layer_name)
        if layer_name not in input_layers.keys():
            input_layers[layer_name] = [layer.name]
        else:
            input_layers[layer_name].append(layer.name)
print(input_layers)

output_tensors[model.layers[0].name] = model.input[0]
output_tensors[model.layers[1].name] = model.input[1]
print(output_tensors)



