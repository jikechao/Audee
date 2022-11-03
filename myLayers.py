from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, \
    GlobalAveragePooling2D, ZeroPadding2D, Flatten, Dense, Activation, ReLU, LeakyReLU, PReLU, ELU, ThresholdedReLU, \
    BatchNormalization, SimpleRNN, LSTM, GRU, Embedding, Dropout, GaussianNoise, ConvLSTM2D, Conv2DTranspose
import numpy as np
np.random.seed(20210808)


def random_layer(layer_name):
    if layer_name == 'Conv2D':
        return Conv2D(filters=np.random.randint(1, 21),
                      kernel_size=(np.random.randint(1,21), np.random.randint(1,21)),
                      strides=(np.random.randint(1,6), np.random.randint(1,6)),
                      padding=np.random.choice(["valid", "same"]),
                      activation=np.random.choice(['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',  'exponential', 'linear']),
                      kernel_initializer=np.random.choice(['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal',  'glorot_uniform', 'he_normal', 'lecun_normal', 'Identity']),)
    elif layer_name == 'DepthwiseConv2D':
        return DepthwiseConv2D(kernel_size=(np.random.randint(1,21),np.random.randint(1,21)),
                               strides=(np.random.randint(1, 6), np.random.randint(1, 6)),
                               padding=np.random.choice(["valid", "same"]),
                               activation=np.random.choice(
                                   ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
                                    'hard_sigmoid', 'exponential', 'linear']),
                               kernel_initializer=np.random.choice(
                                   ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal',
                                    'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform',
                                    'he_normal', 'lecun_normal', 'Identity']),)
    elif layer_name == 'SeparableConv2D':
        return SeparableConv2D(kernel_size=(np.random.randint(1,21),np.random.randint(1,21)),
                               strides=(np.random.randint(1, 6), np.random.randint(1, 6)),
                               padding=np.random.choice(["valid", "same"]),
                               activation=np.random.choice(
                                   ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid',
                                    'hard_sigmoid', 'exponential', 'linear']),
                               kernel_initializer=np.random.choice(
                                   ['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal',
                                    'VarianceScaling', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform',
                                    'he_normal', 'lecun_normal', 'Identity']),)
    elif layer_name == 'MaxPooling2D':
        return MaxPooling2D(pool_size=(np.random.randint(1,21),np.random.randint(1,21)),
                               strides=(np.random.randint(1, 6), np.random.randint(1, 6)),
                               padding=np.random.choice(["valid", "same"]),)
    elif layer_name == 'AveragePooling2D':
        return AveragePooling2D(pool_size=(np.random.randint(1,21),np.random.randint(1,21)),
                               padding=np.random.choice(["valid", "same"]),)
    elif layer_name == 'GlobalMaxPooling2D':
        return GlobalMaxPooling2D(data_format= np.random.choice(['channels_first','channels_last']))
    elif layer_name == 'GlobalAveragePooling2D':
        return GlobalAveragePooling2D(data_format= np.random.choice(['channels_first','channels_last']))
    elif layer_name == 'ZeroPadding2D':
        return ZeroPadding2D(padding=(np.random.randint(1,10), np.random.randint(1,10)),
            data_format= np.random.choice(['channels_first','channels_last']),)
    elif layer_name == 'Flatten':
        return Flatten( data_format= np.random.choice(['channels_first','channels_last']),)
    elif layer_name == 'Dense':
        return Dense(units=np.random.randint(1,101))
    elif layer_name == 'Activation':
        return Activation(activation=np.random.choice(['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',  'exponential', 'linear']))
    elif layer_name == 'ReLU':
        return ReLU(max_value=np.random.random(1)[0],
                    negative_slope=np.random.random(1)[0],
                    threshold=np.random.random(1)[0],)
    elif layer_name == 'LeakyReLU':
        return LeakyReLU(alpha=np.random.random(1)[0],)
    elif layer_name == 'PReLU':
        return PReLU(np.random.choice(['Zeros', 'Ones', 'Constant', 'RandomNormal', 'RandomUniform', 'TruncatedNormal', 'VarianceScaling',
                       'Orthogonal', 'lecun_uniform', 'glorot_normal',  'glorot_uniform', 'he_normal', 'lecun_normal', 'Identity']))
    elif layer_name == 'ELU':
        return ELU(alpha=np.random.random(1)[0])
    elif layer_name == 'ThresholdedReLU':
        return ThresholdedReLU(theta=np.random.random(1)[0])
    elif layer_name == 'BatchNormalization':
        return BatchNormalization(momentum=np.random.random(1)[0],
                                  epsilon=np.random.random(1)[0])
    elif layer_name == 'SimpleRNN':
        return SimpleRNN(units=np.random.randint(1,101),
                         activation=np.random.choice(['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid',  'exponential', 'linear']),
                         dropout=np.random.random(1)[0],
                         recurrent_dropout=np.random.random(1)[0])
    else:
        assert f"OP {layer_name} is not support yet!"


def get_all_changeable_layers(model_name):
    model_api = {}

    model_api['lenet5'] = ['Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Activation', ]
    model_api['resnet20'] = ['Conv2D', 'GlobalAveragePooling2D', 'Dense', 'Activation', 'BatchNormalization', ]
    model_api['vgg16'] = ['Conv2D', 'Flatten', 'Dense', 'Activation', 'BatchNormalization']

    return model_api
