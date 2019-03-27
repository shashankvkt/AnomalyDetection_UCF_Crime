import keras
import scipy.io as sio
import configuration as cfg
import parameters as params
import numpy as np


def build_classifier_model():
    json_file = open(cfg.classifier_model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)
    model = load_weights(model, cfg.classifier_model_weigts)
    return model


def conv_dict(dict2):
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def load_weights(model, weights_file):
    dict2 = sio.loadmat(weights_file)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model


def classify(model, rgb_feature, flow_feature):
    assert len(rgb_feature) == len(flow_feature)
    assert len(rgb_feature) % params.sub_feature_size == 0
    num_sub_features = int(len(rgb_feature)/params.sub_feature_size)
    rgb_sub_features, flow_sub_features = [], []
    for i in range(num_sub_features):
        rgb_sub_features.append(rgb_feature[i*params.sub_feature_size : (i+1)*params.sub_feature_size])
        flow_sub_features.append(flow_feature[i*params.sub_feature_size: (i+1)*params.sub_feature_size])
    new_sub_features = []
    for i in range(num_sub_features):
        new_sub_features.append(np.concatenate([rgb_sub_features[i], flow_sub_features[i]]))

    combined_sub_features = []
    for i in range(0, num_sub_features-1):
        combined_sub_feature = params.lambda_weight * new_sub_features[i] + (1-params.lambda_weight) * new_sub_features[i+1]
        combined_sub_features.append(np.array([combined_sub_feature]))

    prediction = model.predict(combined_sub_features)[0]
    return prediction
