import configuration as cfg
from models.model_i3d import *
from torch import nn


def build_i3d_rgb_feature_extractor_model(use_cuda=False):
    trained_model_file = cfg.i3d_model_rgb
    in_channels = 3
    model = InceptionI3d(num_classes=157, in_channels=in_channels)
    if use_cuda:
        model.load_state_dict(torch.load(trained_model_file))
    else:
        model.load_state_dict(torch.load(trained_model_file, map_location='cpu'))
    return model


def build_i3d_flow_feature_extractor_model(use_cuda=False):
    trained_model_file = cfg.i3d_model_flow
    in_channels = 2
    model = InceptionI3d(num_classes=157, in_channels=in_channels)
    if use_cuda:
        model.load_state_dict(torch.load(trained_model_file))
    else:
        model.load_state_dict(torch.load(trained_model_file, map_location='cpu'))
    return model