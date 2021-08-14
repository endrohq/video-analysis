import os

from flask import Flask, request
from flask_cors import cross_origin

from helpers import download_file
from nlp import get_ner_and_verbs
from transform import get_video_labels

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/files/<cid>/context', methods=['POST'])
@cross_origin()
def annotate_video(cid):

    content = request.json

    ipfs_url = 'https://ipfs.io' + content['ipfsPath']
    temporary_path = os.path.join('temporary')
    local_filename = download_file(ipfs_url, temporary_path)

    local_path = os.path.join('temporary', local_filename)
    # Extract video labels via Pytorch video
    labels = get_video_labels(local_path)

    return {
        "labels": labels
    }


@app.route('/api/annotate', methods=['POST'])
@cross_origin()
def annotate_text():
    content = request.json
    return get_ner_and_verbs(content['description'])

