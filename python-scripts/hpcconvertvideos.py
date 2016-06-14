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
imgDir = "/mnt/earth-alpha/actnet/cvof/"
rgbdir = "/mnt/sun-alpha/actnet/rgb-images/"
smdir = "/mnt/solar-machines/actnet/rgb-images/"
viddir = "/mnt/earth-beta/actnet/videos/"
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

def getVidedInfo(filename):
    
    try:
        cap = cv2.VideoCapture(filename)
    except cv2.error as e:
        print e
        return 0,0,0,0,-1
    if not cap.isOpened(): 
        print "could not open :",filename
        return 0,0,0,0,-1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    return numf,width,height,fps,cap

def getframelabels(videoInfo,numf):
    subset = videoInfo['subset']
    framelabels = np.ones(numf,dtype='uint16')*200;
    if subset == 'testing':
        return framelabels
    else:
        annotations = videoInfo['annotations']
        for annot in annotations:
            actionId = annot['class']
            startframe = annot['sf']
            endframe = annot['ef']
            framelabels[startframe:endframe] = int(actionId)-1
        return framelabels

def checkcvof():
    
    
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
        
    dirlist = os.listdir(rgbdir)
    dirlist = [d for d in dirlist if d.startswith('v_')]
    print 'number of directoires are ',len(dirlist)
    imgsdone = 0;
    vidsdone = 0;
    vcount = 0;
    count = 0;
    for d in dirlist:
        vcount +=1
        if vidsdone>-1:
            savedir = imgDir+d
            # imglist = [img for img in imglist if img.endswith('.jpg')]
            # done = len(imglist)
            if os.path.isdir(savedir):
                vidsdone += 1
                # print 'Vcount ',vcount,' videos done ',vidsdone
            else:
                storageDir = smdir+d+'/';
                    # print 'Vcount ',vcount,' dst ',dst
                if not os.path.isdir(storageDir):
                    count+=1
                    if count>550:
                        t1 = time.time()
                        videoId = d[2:]
                        videoInfo = database[videoId]
                        print 'Doing forward ',d,' count ', count, ' vcount ',vcount
                        videoname = viddir+d+'.mp4'
                        if not os.path.isfile(videoname):
                            videoname = viddir+d+'.mkv'
                        numfs = videoInfo['numf']
                        framelabels = getframelabels(videoInfo,numfs)
                        dst = storageDir+str(numfs-1).zfill(5)+'-ActId'+str(framelabels[-1]).zfill(3)+'.jpg'
                        
                        if not os.path.isfile(dst):
                            numf,width,height,fps,cap = getVidedInfo(videoname)
                            if not cap == -1 and numf == numfs:
                                newW=256;newH=256;
                                framecount = 0;
                                if cap.isOpened():
                                    if not os.path.isdir(storageDir):
                                        os.mkdir(storageDir)
                                    for ind in xrange(numf):
                                        label = framelabels[ind]
                                        dst = storageDir+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                                        retval,image = cap.read()
                                        if not image is None:
                                            resizedImage = cv2.resize(image,(newW,newH))
                                            cv2.imwrite(dst,resizedImage)
                                        else:
                                            cv2.imwrite(dst,resizedImage)
                                            print ' . ',
                                    print dst , 'is created',
                            else:
                                with open('vids/'+videoId+'.txt','wb') as f:
                                    f.write('error')
                        else:
                            print dst,' already there',
                        t2 = time.time()
                        print 'time taken ',t2-t1, ' seconds'
            
if __name__=="__main__":
    checkcvof()
