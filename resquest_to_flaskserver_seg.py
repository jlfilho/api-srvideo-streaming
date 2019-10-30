# importing the requests library
import argparse
import base64
import requests
import json
import numpy as np
import os
import random
import string

from io import BytesIO
from natsort import natsorted
from timeit import default_timer as timer


def randomString(stringLength):
    """Generate a random string with the combination of lowercase and uppercase letters """

    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))



# defining the api-endpoint
API_ENDPOINT = "http://localhost:6006/segment_sr2x/predict/"

# taking input image via command line
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path of the video")
args = vars(ap.parse_args())

video_path = args['video']
b64_video = ""

# List of segments
lsegment = natsorted(os.listdir(video_path))
header = 'Header.m4s'
lsegment.remove(header)
count=0
for segment in lsegment:
    binary_stream = BytesIO()
    with open(video_path+header, "rb") as v:
        byte = v.read(1024)
        while byte:
            binary_stream.write(byte)
            byte = v.read(1024)
    v.close()
    with open(video_path+segment, "rb") as v:
        byte = v.read(1024)
        while byte:
            binary_stream.write(byte)
            byte = v.read(1024)
    v.close()
    binary_stream.seek(0)
    start = timer()
    b64_video = base64.b64encode(binary_stream.read())
    # data to be sent to api
    data = {'b64': b64_video}
    # sending post request and saving response as response object
    r = requests.post(url=API_ENDPOINT, data=data)
    end = timer()
    elapsed = end - start
    count+=1
    print('Seg: {}\nStatus code: {}\nTime elapsed: {}\n'.format(count,r.status_code,elapsed))
    if(r.status_code==200):
        name_seg = randomString(stringLength=15)
        v = open('./video/'+str(count)+'_'+name_seg+'.mp4', "wb")
        segment = base64.b64decode(r.content)
        v.write(segment)
        v.close()
