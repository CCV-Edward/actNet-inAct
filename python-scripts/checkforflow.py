'''
Autor: Gurkirt Singh
Start data: 17th May 2016
purpose: of this file is read annotation from json and
resave annotation with frame level information along with duration info
for eg: a segement described with start frame and end frame
also label it with numeric id rather than text label

'''

import numpy as np
import pickle
import os,time
import shutil
import cv2 as cv2
import time
#subset = 'validation'
subset = 'training'
baseDir = "/mnt/sun-alpha/actnet/";
#imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
imgDir = "/mnt/earth-alpha/actnet/cvof/";
rgbdir = "/mnt/sun-alpha/actnet/rgb-images/"
smdir = "/mnt/solar-machines/actnet/rgb-images/"
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
    
def getNumFrames(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened(): 
        print "could not open :",filename
        return -1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return numf

def getTaxonomyDictionary(taxonomy):
    mytaxonomy = dict();
    for entry in taxonomy:
        nodeName = entry['nodeName'];
        mytaxonomy[nodeName] = entry;
    return mytaxonomy

def checkcvof():
    dirlist = os.listdir(rgbdir)
    dirlist = [d for d in dirlist if d.startswith('v_')]
    print 'number of directoires are ',len(dirlist)
    imgsdone = 0;
    vidsdone = 0;
    vcount = 0;
    for d in dirlist:
        vcount +=1
        if vidsdone>-1:
            savedir = imgDir+d
            # imglist = [img for img in imglist if img.endswith('.jpg')]
            # done = len(imglist)
            if os.path.isdir(savedir):
                vidsdone += 1
                print 'Vcount ',vcount,' videos done ',vidsdone
            else:
                dst = smdir+d+'/';
                print 'Vcount ',vcount,' dst ',dst
                if not os.path.isdir(dst):
                
                    # shutil.rmtree(dst)
                    src = rgbdir+d+'/';
                    t1 = time.time()
                    shutil.copytree(src,dst)
                    t2 = time.time()
                    print 'time taken to copy ',t1-t2
            
if __name__=="__main__":
    checkcvof()
