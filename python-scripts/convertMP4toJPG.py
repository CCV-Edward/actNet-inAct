'''

Autor: Gurkirt Singh
Start data: 2nd May 2016
purpose: of this file is to take all .mp4 videos and convert them to jpg images

'''

import numpy as np
import cv2 as cv2
import os
import shutil
import math

vidDir = "/mnt/sun-alpha/actnet/videos/";
vidDirtemp = "/mnt/earth-beta/Datasets/actnet/videos/";
imgDir = "/mnt/earth-beta/Datasets/actnet/images/";
annotFile = "../anetv13.json"

def getAnnotations():
    with open(annotFile) as f:
        annoData = json.load(f)
    taxonomy = annoData["taxonomy"]
    version = annoData["version"]
    database = annoData["database"]
    print len(database),version,len(taxonomy)
    
def getNumFrames(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened(): 
        print "could not open :",filename
        return -1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return numf

def getVidedInfo(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened(): 
        print "could not open :",filename
        return -1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    return numf,width,height,fps

def getsmallestDimto256(width,height):
    if width>=height:
        newH = 256
        newW = int(math.ceil((float(newH)/height)*width))
    else:
        newW = 256
        newH = int(math.ceil((float(newW)/width)*height))
    return newW,newH

def convertVideos():
    print "this is convertVideos function"
    vidDir = vidDirtemp
    vidlist = os.listdir(vidDir)
    vidlist = [vid for vid in vidlist if vid.endswith(".mp4")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    vcount =0
    for videname in vidlist:
        src = vidDir+videname
        numf,width,height,fps = getVidedInfo(src)
        newW,newH = getsmallestDimto256(width,height)
        print 'old width height were ',width,height,' and newer are ',newW,newH, ' fps ',fps,' numf ', numf, ' vcount  ',vcount
        vcount+=1
        framecount = 0;
        storageDir = imgDir+videname.split('.')[0]+"/"
        imgname = storageDir+str(numf-1).zfill(5)+".jpg"
        if not os.path.isfile(imgname):
            cap = cv2.VideoCapture(filename)
            if cap.isOpened():
                if not os.path.isdir(storageDir):
                    os.mkdir(storageDir)
                for f in xrange(numf):
                    retval,image = cap.read()
                    if not image is None:
                        # print np.shape(retval),np.shape(image), type(image),f
                        resizedImage = cv2.resize(image,(newW,newH))
                        imgname = storageDir+str(framecount).zfill(5)+".jpg"
                        cv2.imwrite(imgname,resizedImage)
                    else:
                        imgname = storageDir+str(framecount).zfill(5)+".jpg"
                        cv2.imwrite(imgname,resizedImage)
                        print 'we have missing frame ',framecount
                    framecount+=1
                print imgname
                
    
def checkConverted():
    print "this is checkConverted videos function"    
    vidlist = os.listdir(vidDir)
    vidlist = [vid for vid in vidlist if vid.endswith(".mp4")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    vcount =0
    for videname in vidlist[15000:]:
        src = vidDir+videname
        numF = getNumFrames(src)
        if numF>0:
            imgname = imgDir+videname.split('.')[0]+"/"+str(numF-1).zfill(5)+".jpg"
            print 'last frame is ',imgname,' vocunt ',vcount
            vcount+=1
            dst = vidDirtemp+videname
            if not os.path.isfile(imgname):
                shutil.move(src,dst)
                print " moved this one to ", dst

if __name__=="__main__":
    # checkConverted()
    convertVideos()
