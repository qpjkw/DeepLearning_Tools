#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:54:38 2018
@author: jiankaiwang
@description: Image Crop
@Reference: https://github.com/jiankaiwang/vott
"""

import argparse
import os
import json
import codecs
import cv2
import logging
import matplotlib.pyplot as plt
import re
import hashlib
import datetime

# In[]

def parseLabelJsonFile(tool="vott", filePath=""):
    """
    description: parse labeling file (must be json format)
    input:
        tool: vott(default) or others
        filePath: the absolute path for the json file
    return :
        json object
    exception:
        1. not json format
        2. unsupported labeling tools
    """
    def parseVottJson(filePath):
        tmp = ""
        with codecs.open(filePath, "r", "utf-8") as fin:
            for line in fin:
                tmp += line.strip()
        try:
            tmp = json.loads(tmp)
            return tmp, 0
        except:
            return "Can not parse json file.", 1
    
    if tool == "vott":
        data, error = parseVottJson(filePath)
        if not error:
            return data
        else:
            raise Exception("Parsing label file was error.")
    else:
        raise Exception("unsupported label type")

# In[]
        
def mapLabel2Imgs(label, imgDir):
    """
    description: find the corresponding between frame id and image file name
    input: 
        label: the label json format from parseLabelJsonFile()
        imgDir: the absolute path for the whole image
    return: 
        { id : file_name }
    """
    allImageFiles = {}
    oriFilenames = next(os.walk(imgDir))[2]
    label_keys = list(label["frames"].keys())
    for name in oriFilenames:
        find_fid = re.search('frame_(\d+)\.jpg', name)
        if find_fid == None:
            continue
        fid = find_fid.groups()[0]
        if fid in label_keys:
            # only when the key is available in the label file
            allImageFiles[fid] = name
    return allImageFiles

# In[]
    
def hashName(algo="SHA1"):
    """
    description: generate hash code based on time
    input: None
    return: a hash string
    """
    hash_object = hashlib.sha1(str(datetime.datetime.now()).encode())
    hex_dig = hash_object.hexdigest()
    return "{}".format(hex_dig)

# In[]
def createFolder(path):
    """
    description: create a folder if it does not exist
    """
    if not os.path.exists(path):
        os.mkdir(path)

# In[]
        
def outputImg(partial_path, img):
    """
    description: save a image
    input:
        partial_path: the upstream folder path
        img: image object
    return: None
    """
    tgtPath = "{}_{}.jpg".format(partial_path, hashName())
    while os.path.exists(tgtPath):
        tgtPath = "{}_{}.jpg".format(partial_path, hashName())
    plt.imsave(tgtPath, img)

# In[]        
def cropAugmOutput(label, label_img_map, imgDir, outputPath, augmentation=True):    
    
    for frame_idx in list(label_img_map.keys()):
        
        label_obj = label["frames"][frame_idx]
        
        # h x w x 3, [b, g, r]
        img = cv2.imread(os.path.join(imgDir, label_img_map[frame_idx]))  
        h, w, c = img.shape
        #print(img.shape)
        
        for obj in label_obj:
            # unit is pixel
            xmin = int((obj["x1"] / obj["width"]) * w)
            ymin = int((obj["y1"] / obj["height"]) * h)
            xmax = int((obj["x2"] / obj["width"]) * w)
            ymax = int((obj["y2"] / obj["height"]) * h)
            tag = obj["tags"][0]
            
            # create folder if it does not exist
            targetFolder = os.path.join(outputPath, tag)
            createFolder(targetFolder)
            
            # region
            ymax = (ymax+1) if (ymax+1) < h else h
            xmax = (xmax+1) if (xmax+1) < w else w
            img_region = img[ymin:ymax, xmin:xmax, :]
            #print(ymin, ymax, xmin, xmax)
            #print(img_region.shape)
            
            # crop and output
            #plt.imshow(img[:,:,::-1])
            
            # sregion image
            region_image = img_region[:,:,::-1]
            #plt.imshow(region_image)
            basename = label_img_map[frame_idx][:-4]
            
            # type
            label_path = "{}/{}/{}".format(outputPath, tag, basename)
                
            # output image
            outputImg(label_path, region_image)

# In[]

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(\
        '--labelfilepath',\
        type=str,\
        default="label.json",\
        help='the label file for image file'\
    )
    parser.add_argument(\
        '--imagedirpath',\
        type=str,\
        default=os.path.join('.','data'),\
        help='the video file path'\
    ) 
    parser.add_argument(\
        '--outputpath',\
        type=str,\
        default=os.path.join('.','crop'),\
        help='path for cropping images'\
    )

    FLAGS, unparsed = parser.parse_known_args()
    
    PATH_LABEL = FLAGS.labelfilepath if os.path.exists(FLAGS.labelfilepath) \
        else "/Users/jiankaiwang/devops/Auxiliary_Operations/data/IMG_1564.m4v.json" 
    PATH_IMG = FLAGS.imagedirpath if os.path.exists(FLAGS.imagedirpath) \
        else "/Users/jiankaiwang/devops/Auxiliary_Operations/data/IMG_1564_frames"
    PATH_CROP = FLAGS.outputpath if os.path.exists(FLAGS.outputpath) \
        else "/Users/jiankaiwang/devops/Auxiliary_Operations/data/IMG_1564_frames_crop"
        
    if not os.path.exists(PATH_CROP):
        os.mkdir(PATH_CROP)
        
    assert os.path.exists(PATH_LABEL), "Label path is not found."
    assert os.path.exists(PATH_IMG), "Image dir is not found."
    assert os.path.exists(PATH_CROP), "Path for cropped images is not found."
    
    # initial
    logging.basicConfig(level=logging.DEBUG)
    
    # parse json data
    LABEL = parseLabelJsonFile(filePath=PATH_LABEL)
    logging.debug(list(LABEL.keys()))
    
    # map label to image
    label_img_map = mapLabel2Imgs(LABEL, PATH_IMG)
    logging.debug(label_img_map)
    
    # crop the data
    cropAugmOutput(LABEL, label_img_map, PATH_IMG, PATH_CROP)
    
    
    
    
    

