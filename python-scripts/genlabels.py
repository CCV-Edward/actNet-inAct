'''
Autor: Gurkirt Singh
Start data: 2nd May 2016
purpose: of this file is read annotation from json and
resave annotation with frame level information along with duration info
for eg: a segement described with start frame and end frame
also label it with numeric id rather than text label

'''

import numpy as np
import pickle
import os
import shutil

import cv2 as cv2
import time
#subset = 'validation'
subset = 'training'

baseDir = "/mnt/sun-alpha/actnet/";
imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
# imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
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

    
def writelabels():
    subset = 'training'
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    ind = np.arange(0,201)     
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    actionframecount = np.zeros(201,dtype='uint32')
    imagelists = [[] for i in range(201)]
    ecount = 0;
    vcount=0;
    
    for videoId in database.keys():
        ecount+=1
        if ecount>=0:
            imgDirV = imgDir+'v_'+videoId+'/'
            videoInfo = database[videoId]
            if not videoInfo['isnull'] and videoInfo['subset'] == subset:
                vcount+=1
                print imgDirV,' ecount ',ecount,videoInfo['subset'],' vcount ',vcount
                numfs = videoInfo['numf']
                framelabels = np.ones(numfs,dtype='uint16')*200;
                annotations = videoInfo['annotations']
                bgcount = numfs;
                for annot in annotations:
                    actionId = annot['class']
                    startframe = annot['sf']
                    endframe = annot['ef']
                    framelabels[startframe:endframe] = int(actionId)-1
                    actionframecount[actionId-1]+=endframe-startframe
                actionframecount[200]+=bgcount;

                for ind,label in enumerate(framelabels):
                    dst = imgDirV+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                    imagelists[label].append(dst)
                    
        
    filename = baseDir+'lists/{}-ImageLists.pkl'.format(subset)
    print 'save imglist in ',filename
    with open(filename,'wb') as f:
        pickle.dump(imagelists,f)
        
def checkmogrify():
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    ind = np.arange(0,201)     
    
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    ecount = 0;
    vcount=0;
    
    for videoId in reversed(database.keys()):
        ecount+=1
        if ecount>=0:
            imgDirV = imgDir+'v_'+videoId+'/'
            videoInfo = database[videoId]
            if not videoInfo['subset'] == 'testing':
                print imgDirV,' ecount ',ecount,videoInfo['subset'],' vcount ',vcount,
                numfs = videoInfo['numf']
                framelabels = np.ones(numfs,dtype='uint16')*200;
                annotations = videoInfo['annotations']
                for annot in annotations:
                    actionId = annot['class']
                    startframe = annot['sf']
                    endframe = annot['ef']
                    framelabels[startframe:endframe] = int(actionId)-1
                    
                label = framelabels[0]
                dst = imgDirV+str(0).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                if os.path.isfile(dst):
                    image = cv2.imread(dst)
                    (w,h,d) = np.shape(image)
                    if w == h and h==256:
                        vcount+=1
                    else:
                        print w,h
                    
        
def mymogrify():
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)     
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];

    ecount = 0;
    vcount=0;
    
    for videoId in reversed(database.keys()):        
        ecount+=1
        if ecount>14:
            videoInfo = database[videoId]
            if not videoInfo['subset'] == 'testing':
                vst = time.time()
                imgDirV = imgDir+'v_'+videoId+'/'
                print imgDirV,' ecount ',ecount,videoInfo['subset'],' vcount ',vcount,
                numfs = videoInfo['numf']
                tempimglist = os.listdir(imgDirV)
                imglist = [imgDirV+img for img in tempimglist if img.endswith('.jpg')]
                dst = imglist[0]
                image = cv2.imread(dst)
                (w,h,d) = np.shape(image)
                if w == h and h==256:
                    vcount+=1
                else:
                    for dst in imglist:
                        image = cv2.imread(dst)
                        resizedimage = cv2.resize(image,(256,256))
                        cv2.imwrite(dst,resizedimage)
                vet  = time.time()
                print ' time taken ', int(vet-vst), ' seconds'
        
    

            
def genimglist(fid,subset,max1,max2):
    maxallowed = max1;
    filename = baseDir+'lists/{}-ImageLists.pkl'.format(subset)
    print 'loading imglist from ',filename
    with open(filename,'rb') as f:
        imagelists = pickle.load(f)

    for action in range(len(imagelists)):
        actionImagelist = imagelists[action]
        if action == 200:
            maxallowed = max2;
        numImages = len(actionImagelist)
        print 'Number of images for action ',str(action).zfill(3),'are ',str(numImages).zfill(8),
        if numImages>maxallowed:
            print ' '
            index = np.asarray(np.arange(numImages),dtype='uint32')
            np.random.shuffle(index)
            writeImgNamestofile(fid,actionImagelist,index[:maxallowed])
        else:
            print ' ok'
            writeImgNamestofile(fid,actionImagelist)

def writeImgNamestofile(fid,imglist,index=[]):
    if len(index)==0:
        for imgname in imglist:
            fid.write(imgname+'\n')
    else:
        for ind in index:
            imgname = imglist[ind]
            fid.write(imgname+'\n')

def genJointList():
    trainlist = baseDir+'lists/{}-{}-tiny.list'.format('train','valid')
    fid = open(trainlist,'wb');
    genimglist(fid,'training',50000,70000)
    genimglist(fid,'validation',25000,35000)
    fid.close()
    
def genlists():
    subset = 'training'
    subset = 'validation'
    trainlist = baseDir+'lists/{}-teenytiny.list'.format(subset)
    fid = open(trainlist,'wb')
    genimglist(fid,subset,10,15)
    
def genertalabels():
    labellist = baseDir+'lists/labels.list'
    fid = open(labellist,'wb');
    for label in range(201):
        fid.write('ActId'+str(label).zfill(3)+'\n')
    
if __name__=="__main__":
    # writelabels()
    # genimglist()
    genlists()
    # genJointList()
    # genertalabels()
    # checkmogrify()
    # mymogrify()
