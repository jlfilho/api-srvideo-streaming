import argparse
import json
import cv2

import numpy as np
import requests
from model import sr_genarator, scale_lr_imgs, unscale_hr_imgs
from tensorflow.keras.preprocessing import image

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
# Preprocessing our input image
img_hr = image.img_to_array(image.load_img(image_path)) 
scale=2
lr_shape = (int(img_hr.shape[1]/scale), int(img_hr.shape[0]/scale)) 
img_lr = cv2.resize(cv2.GaussianBlur(img_hr,(5,5),0),lr_shape, interpolation = cv2.INTER_CUBIC)


img_lr = scale_lr_imgs(img_lr)


# this line is added because of a bug in tf_serving(1.10.0-dev)
img_lr = img_lr.astype('float16')

img_lr = img_lr.reshape(1,img_lr.shape[0],img_lr.shape[1],img_lr.shape[2])
print(img_lr.shape)
img_lr = img_lr.tolist()

#print(type(img_lr))
myobj = {"instances": img_lr}

# sending post request to TensorFlow Serving server
r = requests.post('http://10.0.0.118:8501/v1/models/upscale2x:predict', json=myobj)
print('Status code: ',r.status_code)
pred = json.loads(r.content.decode('utf-8'))
img_sr = np.array(pred['predictions'])

img_sr = unscale_hr_imgs(img_sr)
# Remove batch dimension
img_sr = img_sr.reshape(img_sr.shape[1], img_sr.shape[2], img_sr.shape[3])
image.save_img('teste.png',img_sr)
print(img_sr.shape)

