import base64
import json
import cv2
import sys
import os
from io import BytesIO
from timeit import default_timer as timer

from util import scale_lr_imgs, unscale_hr_imgs, payloader_pre, payloader_pos
from util import start_ffmpeg_reader, start_ffmpeg_writer, randomString   
from util import write_frame, get_video_size, read_frame, return_seg

sys.path.append('../')
from model import create_model, sr_genarator

import numpy as np
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
#from keras.preprocessing import image


# from flask_cors import CORS

app = Flask(__name__)

model=create_model(scale=2)


# Uncomment this line if you are making a Cross domain request
# CORS(app)


# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/single_sr2x/predict/', methods=['POST'])
def single_frame_restoration():
    # Decoding and pre-processing base64 image
    frame_lr = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64']))))
    
    payload = payloader_pre(frame_lr)
    # Making POST request
    r = requests.post('http://10.0.0.118:8501/v1/models/upscale2x:predict', json=payload)
    frame_sr = payloader_pos(r)
    frame_sr = frame_sr.tolist()
    payload = {"predictions": frame_sr}

    # Returning JSON response to the frontend
    return jsonify(payload)



@app.route('/segment_sr2x/predict/', methods=['POST'])
def segment_frame_restoration():
    filename = './'+randomString()+'.mp4'
    filename2 = './'+randomString()+'.mp4' 
    v = open(filename, "wb")
    segment = base64.b64decode(request.form['b64'])
    v.write(segment)
    #v.seek(0)
    v.close()
    width, height = get_video_size(filename)
    print(width, height)
    reader = start_ffmpeg_reader(filename)
    writer = start_ffmpeg_writer(filename2,width*2, height*2)
    time_elapsed = 0
    while True:
        start = timer()
        in_frame = read_frame(reader, width, height)
        if in_frame is None:
            break
        frame_sr = sr_genarator(model,in_frame)
        write_frame(writer, frame_sr)
        
        end = timer()
        elapsed = end - start
        time_elapsed +=(elapsed)    
        
        print('Time per frame: {}'.format(elapsed))
        print('LR frame shape: {}'.format(in_frame.shape))
        print('SR frame shape: {}'.format(frame_sr.shape))

        # payload = payloader_pre(in_frame)
        # # Making POST request
        # r = requests.post('http://10.0.0.118:8501/v1/models/upscale2x:predict', json=payload)
        # print('Status code: {}'.format(r.status_code))
        # if(r.status_code==200):
        #     frame_sr = payloader_pos(r)
        
    print('Time elapsed: {}'.format(time_elapsed))
    reader.wait()
    writer.stdin.close()
    writer.wait()
    os.remove(filename)
    
    return return_seg(filename2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006)