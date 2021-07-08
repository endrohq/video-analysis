import torch

import json
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo, CenterCropVideo
)

from constants import mean, std, side_size, crop_size, num_frames, alpha, model_name, sampling_rate, frames_per_second, device

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


class PackPathway(torch.nn.Module):
    """
    Transform for converting videos frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)


def get_video_labels(video_path):
    # Pick a pretrained model and load the pretrained weights
    model = torch.hub.load('facebookresearch/pytorchvideo', model=model_name, pretrained=True)

    model = model.to(device)
    model = model.eval()

    with open("data/kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=0, end_sec=clip_duration)

    # Apply a transform to normalize the videos input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]

    # Pass the input clip through the model
    preds = model(inputs)

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices

    # Map the predicted classes to the label names
    return [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]


