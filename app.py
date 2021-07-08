
import os
import uuid

import cv2

from flask import Flask, send_from_directory, request
from flask_cors import cross_origin

from nlp import get_ner_and_verbs
from transform import get_video_labels

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/cdn/videos/<id>', methods=['GET'])
@cross_origin()
def retrieve_video(id):
    video_path = id + '.mp4'
    return send_from_directory('./storage/videos', video_path)


@app.route('/cdn/images/<id>', methods=['GET'])
@cross_origin()
def retrieve_image(id):
    video_path = id + '.jpg'
    return send_from_directory('./storage/images', video_path)


@app.route('/process', methods=['POST'])
@cross_origin()
def process():

    media_id = str(uuid.uuid4())
    video_id = media_id + '.mp4'
    image_id = media_id + '.jpg'

    # 'UPLOAD' videos
    file = request.files['file']
    video_path = os.path.join('./storage/videos', video_id)
    file.save(video_path)

    # 'UPLOAD' first frame for video previews
    image_path = os.path.join('./storage/images', image_id)
    cap = cv2.VideoCapture(video_path)
    success,image = cap.read()
    if success:
        cv2.imwrite(image_path, image)

    # Extract video labels via Pytorch video
    labels = get_video_labels(video_path)

    return {
        "mediaId": media_id,
        "labels": labels
    }


@app.route('/annotate', methods=['POST'])
@cross_origin()
def annotate_text():
    content = request.json
    return get_ner_and_verbs(content['description'])

