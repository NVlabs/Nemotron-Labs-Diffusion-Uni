"""One-request-at-a-time HTTP transport; JSON metadata and lossless PNG images."""
import base64
import io
import json
import urllib.request
from PIL import Image


def encode_image(image):
    stream = io.BytesIO(); image.save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


def decode_image(value):
    return Image.open(io.BytesIO(base64.b64decode(value))).convert('RGB')


def request(url, payload, timeout=1800):
    req = urllib.request.Request(url + '/infer', json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()
