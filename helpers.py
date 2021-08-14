import os
import shutil

import requests


def download_file(url, folder_name):
    local_filename = url.split('/')[-1]
    path = os.path.join(folder_name, local_filename)
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename