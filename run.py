import keras
import myLayers
import numpy as np
import logging
np.random.seed(20210808)

logging.basicConfig(level=logging.INFO, filename='mylog.log')

# seed_model_path = 'D:\server-backup-137\data\keras_model\lenet5-mnist_origin.h5'
seed_model_path = 'D:\server-backup-137\data\keras_model\lenet5-fashion-mnist_origin.h5'
seed_model_path = 'D:\server-backup-137\data\keras_model\\vgg16-cifar10_origin.h5'
seed_model = keras.models.load_model(seed_model_path)
seed_model.summary()


def random_para_layer(layer, random_layer):
    new_config = layer.get_config()
    for para, v in random_layer.get_config().items():
        if para == 'name':
            continue
        new_config[para] = v
        # print(para, v)
    new_config['name'] = new_config['name'] + '_random'
    # print(new_config['name'])
    new_layer = layer.__class__.from_config(new_config)
    return new_layer


def select_one_layer(model_name):
    model_api = myLayers.get_all_changeable_layers(model_name)
    related_api = []
    if 'lenet5' in model_name:
        related_api = model_api[model_name]
    elif 'resnet20' in model_name:
        related_api = model_api[model_name]
    elif 'vgg16' in model_name:
        related_api = model_api[model_name]
    else:
        logging.error(f"{model_name} is not support yet!")
    selected_api_name = np.random.choice(related_api)
    selected_api = myLayers.random_layer(selected_api_name)
    return selected_api


def layer_copy(layer):
    new_layer = layer.__class__.from_config(layer.get_config())
    if np.shape(layer.get_weights()) == (0,):  # such as max pooling, flatten,dropout, no weights.
        print(layer)
        new_layer.set_weights(layer.get_weights())
        print(np.shape(layer.get_weights()))
        # print(layer.name+'no weight')
    elif np.shape(layer.get_weights()) == new_layer.get_weights():
        new_layer.set_weights(layer.get_weights())
    else:
        new_layer.set_weights(np.random.random(np.shape(new_layer.get_weights())))
    return new_layer


def gen_new_model(seed_model):
    new_layers = []
    finish_mutated_flag = False
    model_mode = seed_model.__class__.__name__
    model_type_name = seed_model_path.split('\\')[-1].split('-')[0]
    if model_mode == 'Sequential':
        while not finish_mutated_flag:
            new_layers = []
            changed_api = select_one_layer(model_type_name)
            changed_api_name = changed_api.name
            for i, layer in enumerate(seed_model.layers):
                layer_type_name = layer.name[:-1 - len(layer.name.split('_')[-1])]
                if not finish_mutated_flag and layer_type_name == changed_api_name:
                    new_layer = random_para_layer(layer, changed_api)
                    finish_mutated_flag = True
                else:
                    new_layer = layer_copy(layer)
                new_layers.append(new_layer)
        new_model = keras.Sequential(layers=new_layers, name=seed_model.name + '-new')

        new_model.build(seed_model.input_shape)
        return new_model
    else:
        operation= None
        suffix=None
        input_layers = {}
        output_tensors = {}
        model_output = None
        for layer in seed_model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in input_layers.keys():
                    input_layers[layer_name] = [layer.name]
                else:
                    input_layers[layer_name].append(layer.name)

        output_tensors[seed_model.layers[0].name] = seed_model.input

        for layer in seed_model.layers[1:]:
            layer_input_tensors = [output_tensors[l] for l in input_layers[layer.name]]
            if len(layer_input_tensors) == 1:
                layer_input_tensors = layer_input_tensors[0]

            if operation is not None and layer.name in operation.keys():
                x = layer_input_tensors
                cloned_layer = layer_copy(layer)
                if suffix is not None:
                    cloned_layer.name += suffix
                x = operation[layer.name](x, cloned_layer)
            else:
                cloned_layer = layer_copy(layer)
                if suffix is not None:
                    cloned_layer.name += suffix
                x = cloned_layer(layer_input_tensors)

            output_tensors[layer.name] = x
            model_output = x
        return keras.Model(inputs=seed_model.inputs, outputs=model_output)


new_model = gen_new_model(seed_model)
new_model.save("m2.h5")
m_ = keras.models.load_model('m2.h5')
m_.summary()

#
# input = np.random.random((2, 28, 28, 1))
# input = np.round(input * 255)
# # print(input)
# res = m_.predict(input)
# print(res)
# res2 = seed_model.predict(input)
# print(res2)
