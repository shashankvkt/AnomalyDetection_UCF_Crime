from utils.video_util import *
from utils.array_util import *
from i3d import *
from flownet import *
from classifier import *


def run_demo():
    # read video
    video_clips = get_video_clips(cfg.sample_video_path)

    print("Number of clips in the video : ", len(video_clips))

    use_cuda = False
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        use_cuda = True

    # load models
    i3d_rgb_model = build_i3d_rgb_feature_extractor_model(use_cuda=use_cuda)
    i3d_flow_model = build_i3d_flow_feature_extractor_model(use_cuda=use_cuda)
    flownet_model = build_flownet_model(use_cuda)
    classifier_model = build_classifier_model()

    print("Models initialized")

    # push to GPU if available
    if use_cuda:
        i3d_rgb_model.cuda(device=0)
        i3d_flow_model.cuda(device=0)
        flownet_model.model.cuda(device=0)

    # iterate through all clips and extract RGB and OpticalFlow features
    rgb_features, flow_features = [], []
    for i, clip in enumerate(video_clips):
        if len(clip) < params.frame_count:
            continue
        print("Processing clip : ", i, end='\r')
        clip_tensor = np.transpose([clip], (0, 4, 1, 2, 3))
        clip_tensor = Variable(torch.from_numpy(clip_tensor).float(), requires_grad=False)
        if use_cuda:
            clip_tensor = clip_tensor.cuda(device=0)
        rgb_feature = i3d_rgb_model.extract_features(clip_tensor).cpu().data.squeeze().numpy()
        if num_devices > 0:
            flow_clip = get_opticalflow_clip(flownet_model, clip, device=0)
        else:
            flow_clip = get_opticalflow_clip(flownet_model, clip, device=None)
        flow_clip_tensor = np.transpose([flow_clip], (0, 4, 1, 2, 3))
        flow_clip_tensor = Variable(torch.from_numpy(flow_clip_tensor).float(), requires_grad=False)
        if use_cuda:
            flow_clip_tensor = flow_clip_tensor.cuda(device=0)
        flow_feature = i3d_flow_model.extract_features(flow_clip_tensor).cpu().data.squeeze().numpy()
        rgb_features.append(rgb_feature)
        flow_features.append(flow_feature)

    rgb_features = np.array(rgb_features)
    flow_features = np.array(flow_features)

    # bag rgb and flow features
    rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)
    flow_feature_bag = interpolate(flow_features, params.features_per_bag)

    assert len(rgb_feature_bag) == len(flow_feature_bag)

    # classify using the trained classifier model
    predictions = []
    for i in range(params.features_per_bag):
        rgb_feature = rgb_feature_bag[i]
        flow_feature = flow_feature_bag[i]
        prediction = classify(classifier_model, rgb_feature, flow_feature)
        predictions.append(prediction)

    predictions = np.array(predictions)

    # visualize the results
    print(predictions.shape)


if __name__ == '__main__':
    run_demo()