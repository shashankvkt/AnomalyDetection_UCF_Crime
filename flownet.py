import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import configuration as cfg
from models import flow_transforms
from models.FlowNetS import *

warnings.filterwarnings('ignore')


class FlowNetOFExtractor:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.model = self.build_model()

    def build_model(self):
        # Build model
        if self.use_cuda:
            trained_model = torch.load(cfg.flownet_model)
        else:
            trained_model = torch.load(cfg.flownet_model, map_location='cpu')
        model = flownets(trained_model)
        model.eval()
        return model

    def extract(self, img1, img2, device):
        # B x 3(RGB) x 2(pair) x H x W
        input_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        ])
        img1 = input_transform(img1)
        img2 = input_transform(img2)
        inputs = torch.cat([img1, img2]).unsqueeze(0)

        if device is not None:
            inputs = Variable(inputs, requires_grad=False).cuda(device=device)
        else:
            inputs = Variable(inputs, requires_grad=False)

        output = self.model(inputs)

        output = F.interpolate(output, size=img1.size()[-2:], mode='bilinear', align_corners=False)

        output = np.transpose(output.cpu().data.numpy().squeeze(), (1, 2, 0))
        return output


def build_flownet_model(use_cuda=False):
    opticalflow_extractor = FlowNetOFExtractor(use_cuda)
    return opticalflow_extractor


def get_opticalflow_clip(opticalflow_extractor, rgb_clip, device):
    num_frames = len(rgb_clip)
    opticalflow_clip = []
    reference_index = 0
    for i in range(num_frames):
        opticalflow_frame = opticalflow_extractor.extract(rgb_clip[reference_index], rgb_clip[i], device)
        opticalflow_clip.append(opticalflow_frame)
    return np.array(opticalflow_clip)

