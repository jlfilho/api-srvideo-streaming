import logging
import subprocess
import ffmpeg
import json
import base64
import random
import string
import os

import numpy as np
from io import BytesIO


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def randomString(stringLength=15):
    """Generate a random string with the combination of lowercase and uppercase letters """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))


def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def start_ffmpeg_reader(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_writer(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_frame(reader, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = reader.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def process_frame_simple(frame):
    '''Simple processing example: darken frame.'''
    return frame * 1.

def write_frame(writer, frame):
    logger.debug('Writing frame')
    writer.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to ESRGAN"""
        return imgs / 255.

def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        imgs = (imgs + 1.) * 127.5
        imgs = np.clip(imgs, 0., 255.)
        return imgs.astype('uint8')

def payloader_pre(frame_lr):
    # Scaling in range [0,1]
    frame_lr = scale_lr_imgs(frame_lr)
    # Reshaping frame 4-dim for use with tensorflow
    frame_lr = frame_lr.reshape(1,frame_lr.shape[0],frame_lr.shape[1],frame_lr.shape[2])
    # Converting to list to serialize object
    frame_lr = frame_lr.tolist()
    # Creating payload for TensorFlow serving request
    payload = {"instances": frame_lr}
    return payload

def payloader_pos(r):
    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('latin-1', errors="replace"))
    # Converting to numpy array
    frame_sr = np.array(pred['predictions'])
    # Uniscaling in range of [0,255]
    frame_sr = unscale_hr_imgs(frame_sr)
    # Removing batch dimension
    frame_sr = frame_sr.reshape(frame_sr.shape[1], frame_sr.shape[2], frame_sr.shape[3])
    return frame_sr

def return_seg(video_path):
    binary_stream = BytesIO()
    with open(video_path, "rb") as v:
        byte = v.read(1024)
        while byte:
            binary_stream.write(byte)
            byte = v.read(1024)
    v.close()
    os.remove(video_path)
    binary_stream.seek(0)
    b64_video = base64.b64encode(binary_stream.read())
    return b64_video
