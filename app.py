import json
import os
import uuid

import torch
from flask import Flask, send_from_directory, request
from flask_cors import cross_origin
from pytorchvideo.data.encoded_video import EncodedVideo
from werkzeug.utils import secure_filename

from transform import transform
from constants import model_name, num_frames, sampling_rate, frames_per_second, device

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/cdn/<id>', methods=['GET'])
@cross_origin()
def download_file(id):
    return send_from_directory(app.config["UPLOAD_FOLDER"], id)


@app.route('/cdn', methods=['POST'])
@cross_origin()
def upload():

    video_id = str(uuid.uuid4())

    # 'UPLOAD' video
    file = request.files['file']
    filename = secure_filename(file.filename)
    video_path = os.path.join('./storage', video_id)
    file.save(video_path)

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

    # Apply a transform to normalize the video input
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
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]

    return {
        "labels": pred_class_names,
        "videoId": video_id
    }
