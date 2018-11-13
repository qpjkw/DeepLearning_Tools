#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:07:42 2018
@author: jiankaiwang
@description: data augementation
@Reference: https://sophia.ddns.net/data/ImageDataAugmentation.html
"""

import argparse
import os
import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import datetime

# In[]
    
def hashName(algo="SHA1"):
    """
    description: generate a hash string
    input: None
    return: a hash string
    """
    hash_object = hashlib.sha1(str(datetime.datetime.now()).encode())
    hex_dig = hash_object.hexdigest()
    return "{}".format(hex_dig)

# In[]

def outputImg(partial_path, img):
    """
    description: save a image
    input:
        partial_path: the absolute path of the image dir
        img: image object
    return: None
    """
    tgtFile = "{}/{}.jpg".format(partial_path, hashName())
    while os.path.exists(tgtFile):
        tgtFile = "{}/{}.jpg".format(partial_path, hashName())
    plt.imsave(tgtFile, img)
    
# In[]
    
def listFiles(inpath):
    """
    description: filter the image by its extension name
    input: the searching dir
    output: a list with all validated image files
    """
    allowedFiles = ["jpg","JPG","jpeg","JPEG","png","PNG"]
    files = []
    for file in next(os.walk(inpath))[2]:
        tmpList = file.split(".")
        if tmpList[-1] in allowedFiles:
            files.append(file)
    return files
    
# In[]

def doflip(inpath, outpath):
    """
    description: flip the image
    """
    def flip(img, opt="hor"):
        if opt == "hor":
            return cv2.flip(img, 1)
        elif opt == "ver":
            return cv2.flip(img, 0)
        else:
            return cv2.flip(img, -1)
        
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        outputImg(outpath, flip(img, "hor"))
        outputImg(outpath, flip(img, "ver"))
        outputImg(outpath, flip(img, "both"))

def dobrightness(inpath, outpath):
    """
    description: change image brightness
    """
    def AdjBrightness(image, bright=0):
        # RGB -> HSV (Hue, Saturation, Value)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.uint16)
        hsv[:,:,2] += bright
        hsv[:,:,2] = np.minimum(hsv[:,:,2], 255)
        hsv[:,:,2] = np.maximum(hsv[:,:,2], 0)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        outputImg(outpath, AdjBrightness(img, bright=10))
        #outputImg(outpath, AdjBrightness(img, bright=30))
        outputImg(outpath, AdjBrightness(img, bright=50))
        #outputImg(outpath, AdjBrightness(img, bright=70))    
        outputImg(outpath, AdjBrightness(img, bright=90))

def docontrast(inpath, outpath):
    """
    description: change image contrast
    """
    def contrast(image, value=3.0):
        # LAB channel 
        lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
        # Splitting the LAB image to different channels
        l, a, b = cv2.split(lab)
    
        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=value, tileGridSize=(8,8))
        cl = clahe.apply(l)
    
        # Merge the CLAHE enhanced L-channel with the a and b channel
        clab = cv2.merge((cl,a,b))
        
        return cv2.cvtColor(clab, cv2.COLOR_LAB2BGR)
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        outputImg(outpath, contrast(img, value=0.5))
        outputImg(outpath, contrast(img, value=1.0))
        #outputImg(outpath, contrast(img, value=1.5))    

def dotone(inpath, outpath):
    """
    description: change image tone
    """
    def tone(image, color_value=0):
        # RGB -> HSV (Hue, Saturation, Value)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.uint16)
        hsv[:,:,0] += color_value
        hsv[:,:,0] = np.minimum(hsv[:,:,0], 255)
        hsv[:,:,0] = np.maximum(hsv[:,:,0], 0)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        outputImg(outpath, tone(img, 10))
        outputImg(outpath, tone(img, 20))
        outputImg(outpath, tone(img, 30)) 

def docrop(inpath, outpath):
    """
    description: random crop the image into small sections
    """
    def crop(image, scale=0.5):
        (h, w) = image.shape[:2]
        new_h = int(h * 0.5)
        new_w = int(w * 0.5)
        rand_x = np.random.randint(w * (1-scale))
        rand_y = np.random.randint(h * (1-scale))
        #print(h, w, new_h, new_w, rand_x, rand_y)
        return image[rand_y : (rand_y + new_h), rand_x : (rand_x + new_w)]
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        outputImg(outpath, crop(img, 0.95))
        outputImg(outpath, crop(img, 0.95))
        #outputImg(outpath, crop(img, 0.95))

def doshift(inpath, outpath):
    """
    description: shift small image section up/down/left/right
    """
    def shift(image, x, y):
        m = np.float32([[1,0,x],[0,1,y]])
        return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        outputImg(outpath, shift(img, 0, 3))
        outputImg(outpath, shift(img, 0, -3))
        outputImg(outpath, shift(img, 3, 0))
        outputImg(outpath, shift(img, 0, -3))

def dorotate(inpath, outpath):
    """
    description: rotate image section
    """
    def rotate(image, angle=0, center=None, scale=1.0):
        (h, w) = image.shape[:2]   # get width and height
        
        if center is None:
            # get the image center as the default center
            center = (w / 2, h / 2)
        
        # get rotation mapping matrix
        m = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, m, (w, h))
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        #outputImg(outpath, rotate(img, 45))
        #outputImg(outpath, rotate(img, -45))
        outputImg(outpath, rotate(img, 90))
        outputImg(outpath, rotate(img, -90))
        outputImg(outpath, rotate(img, 180))
        outputImg(outpath, rotate(img, -180))


def doscale(inpath, outpath):
    """
    description: scale or resize image
    """
    def resize(image, width=None, height=None, method=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        
        if width is None and height is None:
            return image
        
        # calculate the ratio
        if width is None:
            ratio = float(height) / float(h)
            dim = (int(ratio * w), int(height))
        else:
            ratio = float(width) / float(w)
            dim = (int(width), int(h * ratio))
        
        # resize the image
        resized = cv2.resize(image, dim, interpolation=method)
        
        return resized
    
    files = listFiles(inpath)
    for file in files:
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(inpath, file))
        img = img[:,:,::-1]
        #outputImg(outpath, rotate(img, 45))
        #outputImg(outpath, rotate(img, -45))
        outputImg(outpath, resize(img, width=img.shape[1]*2))
        outputImg(outpath, resize(img, width=img.shape[1]*0.5))

# In[]

if __name__ == "__main__":
    
    """
    augmentation: 
        Flip, Brightness, Contrast, Tone, Crop, Translate, Rotation, Scale
          1       1          1        0    0       1           1       0
    a = 2 ** np.array(list(range(0,8)))
    b = np.array([1, 1, 1, 0, 0, 1, 1, 0])
    flag = np.sum(a * b)
    (default = 103)
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--imagedirpath',\
        type=str,\
        default=os.path.join('.','data'),\
        help='files for argumentation'\
    ) 
    parser.add_argument(\
        '--outputpath',\
        type=str,\
        default=os.path.join('.','crop'),\
        help='path for cropping images'\
    )
    parser.add_argument(\
        '--augmentation',\
        type=int,\
        default=103,\
        help='data augmentation'\
    )
    FLAGS, unparsed = parser.parse_known_args()
    
    PATH_IMG = FLAGS.imagedirpath if os.path.exists(FLAGS.imagedirpath) \
        else "/Users/jiankaiwang/devops/Auxiliary_Operations/data/IMG_1564_frames_crop/hole"
    PATH_ARG = FLAGS.outputpath if os.path.exists(FLAGS.outputpath) \
        else "/Users/jiankaiwang/devops/Auxiliary_Operations/data/IMG_1564_frames_crop/hole"
    DATA_AUGM = FLAGS.augmentation
        
    assert os.path.exists(PATH_IMG), "Image dir is not found."
    assert os.path.exists(PATH_ARG), "Path for cropped images is not found."
    
    # initial
    logging.basicConfig(level=logging.DEBUG)
    
    # file status
    initialFileCounts = len(listFiles(PATH_IMG))
    print("total {} images in initial".format(initialFileCounts))
    
    allopts = list(reversed("{0:b}".format(int(DATA_AUGM))))
    if len(allopts) < 8:
        allopts = allopts + ['0'] * (8 - len(allopts))
    allopts = np.array(allopts, dtype=int)
    
    # start to augument images
    if allopts[0] == 1:
        doflip(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Flip operation finished.")
    
    if allopts[1] == 1:
        dobrightness(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Brightness operation finished.")
    
    if allopts[2] == 1:
        docontrast(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Contrast operation finished.")
        
    if allopts[3] == 1:
        dotone(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Tone operation finished.")
    
    if allopts[4] == 1:
        docrop(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Crop operation finished.")
    
    if allopts[5] == 1:
        doshift(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Shift operation finished.")
    
    if allopts[6] == 1:
        dorotate(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Rotate operation finished.")
        
    if allopts[7] == 1:
        doscale(PATH_IMG, PATH_ARG)
        print("DATA_AUG: Scale operation finished.")
    
    # final status
    finalFileCounts = len(listFiles(PATH_ARG))
    print("total {} images in final".format(finalFileCounts))
    