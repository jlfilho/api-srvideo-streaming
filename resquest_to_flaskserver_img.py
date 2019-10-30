# importing the requests library
import argparse
import base64
import requests
import json
import numpy as np

from tensorflow.keras.preprocessing import image

# defining the api-endpoint
API_ENDPOINT = "http://localhost:6006/single_sr2x/predict/"

# taking input image via command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
b64_image = ""
# Encoding the JPG,PNG,etc. image to base64 format
with open(image_path, "rb") as imageFile:
    b64_image = base64.b64encode(imageFile.read())

# data to be sent to api
data = {'b64': b64_image}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=data)

print('Status code: ',r.status_code)

pred = json.loads(r.content.decode('utf-8'))
img_sr = np.array(pred['predictions'])
image.save_img('teste.png',img_sr)
print(img_sr.shape)