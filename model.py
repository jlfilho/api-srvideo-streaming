import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES']='0' #Set a single gpu
warnings.filterwarnings("ignore")

import tensorflow as tf

import numpy as np
import skvideo.io

#from tensorflow import keras

from argparse import ArgumentParser
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Add, Conv2D, Dense, Lambda, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras.initializers import VarianceScaling, RandomNormal
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.regularizers import l1,l2,l1_l2
from tqdm import tqdm
from timeit import default_timer as timer

l1_reg = l1(1e-3)
l2_reg = l2(1e-3)
l1l2_reg = l1_l2(1e-3)



def load_weights(model=None,weights=None,**kwargs):
        print(">> Loading weights...")
        if weights:
            model.load_weights(weights, **kwargs)


def save_model(model=None):
        """Save the model networks"""
        print(">> Saving model...")
        export_path = './model/1'
        model.save(export_path)
        

def compile_model(model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss="mse",
            optimizer=Adam(lr=1e-3,beta_1=0.9, beta_2=0.999)
        )
        

        
def rtvsrgan(channels=3,scale=2):
    varscale = 1.
        
    def SubpixelConv2D(scale=2,name="subpixel"):
        
        def subpixel_shape(input_shape):
            dims = [input_shape[0],
                    None if input_shape[1] is None else input_shape[1] * scale,
                    None if input_shape[2] is None else input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            output_shape = tuple(dims)
            return output_shape

        def subpixel(x):
            return tf.nn.depth_to_space(x, scale)

        return Lambda(subpixel, output_shape=subpixel_shape, name=name)

    inputs = Input(shape=(None, None, channels),name='input_1')

    x = Conv2D(filters = 32, kernel_size = (3,3), strides=1, kernel_regularizer=l2_reg, 
            kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
            padding = "same",name='conv_1')(inputs) 
    x = LeakyReLU(0.2)(x) 
    x1 = x

    x = Conv2D(filters = 32, kernel_size = (3,3), strides=1, kernel_regularizer=l2_reg, 
            kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
            padding = "same",name='conv_2')(x) 
    x2 = LeakyReLU(0.2)(x)
    x = Concatenate()([x1, x2])

    x = Conv2D(filters = 32, kernel_size = (3,3), strides=1, kernel_regularizer=l2_reg, 
            kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
            padding = "same",name='conv_3')(x) 
    x = LeakyReLU(0.2)(x)

    x = Lambda(lambda x: x * 0.2)(x)
    x = Concatenate()([x1, x2,x])

    
    x = Conv2D(filters = scale**2*channels, kernel_size = (3,3), strides=1, kernel_regularizer=l2_reg, 
            kernel_initializer=VarianceScaling(scale=varscale, mode='fan_in', distribution='normal', seed=None),
            padding = "same", name='conv_4')(x) 
    
    x = SubpixelConv2D(scale=scale,name='subpixel_')(x)

    x = Activation('tanh',name='tanh_')(x) 

    model = Model(inputs=inputs, outputs=x)
    #model.summary()
    return model


def scale_lr_imgs(imgs):
        """Scale low-res images prior to passing to ESRGAN"""
        return imgs / 255.


def unscale_hr_imgs(imgs):
        """Un-Scale high-res images"""
        imgs = (imgs + 1.) * 127.5
        imgs = np.clip(imgs, 0., 255.)
        return imgs.astype('uint8')

def sr_genarator(model,img_lr):
        """Predict sr frame given a LR frame"""
        # Predict high-resolution version (add batch dimension to image)
        img_lr = scale_lr_imgs(img_lr)
        #img_sr = model.predict(np.expand_dims(img_lr, 0))
        img_sr = model.predict(img_lr.reshape(1,img_lr.shape[0],img_lr.shape[1],img_lr.shape[2]))
        img_sr = unscale_hr_imgs(img_sr)
        # Remove batch dimension
        img_sr = img_sr.reshape(img_sr.shape[1], img_sr.shape[2], img_sr.shape[3])
        return img_sr

def selectBetterBitrate(height, fps):   
        print(height,fps)
        if (height < 200):
                bitrate = "280k"
        elif (height < 300):
                bitrate = "400k"    
        elif((height < 400) and (fps < 40)):
                bitrate = "1M"
        elif((height < 400) and (fps < 70)):
                bitrate = "1.5M"
        elif((height < 500) and (fps < 40)):
                bitrate = "2.5M"
        elif((height < 500) and (fps < 70)):
                bitrate = "4M"
        elif((height < 800) and (fps < 40)):
                bitrate = "5M"
        elif((height < 800) and (fps < 70)):
                bitrate = "7.5M"
        elif((height < 1100) and (fps < 40)):
                bitrate = "8M"
        elif((height < 1100) and (fps < 70)):
                bitrate = "12M"
        elif((height < 1600) and (fps < 40)):
                bitrate = "16M"
        elif((height < 1600) and (fps < 70)):
                bitrate = "24M"
        elif((height < 2400) and (fps < 40)):
                bitrate = "40M"
        elif((height < 2400) and (fps < 70)):
                bitrate = "55M"
        elif((height < 3000) and (fps < 40)):
                bitrate = "50M"
        elif((height < 3000) and (fps < 70)):
                bitrate = "70M"
        else:
                print(">> Unknow resolution.")
                exit()
        print(">> BITRATE: ",bitrate)
        return bitrate


def write_srvideo(model=None,lr_videopath=None,sr_videopath=None,scale=None,print_frequency=False,crf=15,fps=None,gpu=False):
        """Generate SR video given LR video """
        videogen = skvideo.io.FFmpegReader(lr_videopath)
        t_frames, height, width, _  = videogen.getShape() 
        print(">> Inputshape: ",videogen.getShape())
        metadata = skvideo.io.ffprobe(lr_videopath)
        #print(json.dumps(metadata["video"], indent=4))
        _fps = metadata['video']['@r_frame_rate'] if (fps == None) else str(fps)
        
        codec = 'h264_nvenc' if (gpu == 'True') else 'libx264' 
        writer = skvideo.io.FFmpegWriter(sr_videopath, 
        inputdict={'-r': _fps, '-width': str(width*scale), '-height': str(height*scale)},
        outputdict={'-vcodec': codec, '-r': _fps, '-crf': str(crf), '-pix_fmt': 'yuv420p',
        '-b:v': selectBetterBitrate(height*scale,int(_fps.split('/')[0])/int(_fps.split('/')[1]))})
        count = 0
        time_elapsed = 0
        print(">> Writing video...")
        for frame in tqdm(videogen):
                #frame = downsample(frame,scale)
                start = timer()
                img_sr = sr_genarator(model,frame)
                writer.writeFrame(img_sr)
                end = timer()
                elapsed = end - start
                time_elapsed +=(elapsed)
                count +=1
                if (print_frequency): 
                        if(count % print_frequency == 0):
                                print("")
                                print('... Time per Frame: '+str(elapsed)+'s')
                                print('... FPS: '+str(1./elapsed))
                                #print('... Estimated time: '+str(np.mean(time_elapsed)*(t_frames-count)/60.)+'min')
        writer.close()
        videogen = skvideo.io.FFmpegReader(sr_videopath)
        print(">> Outputshape: ",videogen.getShape())
        print('>> Video resized in '+str(time_elapsed)+'s')
        print('>> Average FTP '+str(count/time_elapsed))
        return time_elapsed



def parse_args():
    parser = ArgumentParser(description='SR with model RT-VSRGAN')

    parser.add_argument(
        '-s', '--scale',
        type=int, default=2,
        help='Number of upscaling factor.',
    )
 
        
    return  parser.parse_args()

def create_model(scale=2):
    rtvsr = rtvsrgan(channels=3,scale=scale)
    namemodel = '../model/{}_{}X.h5'.format('RTVSRGAN',scale)
    load_weights(model=rtvsr,weights=namemodel)
    return rtvsr 


# Run the RT-VSRGAN network
if __name__ == "__main__":

        args = parse_args()
        print(">> Creating the RT-VSRGAN network")
        rtvsr=create_model(scale=2)
        #compile_model(rtvsr)
        rtvsr.save('./model/M_RTVSRGAN_2X.h5')
        save_model(model=rtvsr)
        path= "./video/mobile_cif.264"
        path2= "./video/mobile_cif.mp4"
        #time_elapsed =  write_srvideo(model=rtvsr,lr_videopath=path,sr_videopath=path2,scale=2,print_frequency=False,crf=15,fps=None,gpu=False)


        