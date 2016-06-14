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
#imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
srcimgDir = "/mnt/earth-alpha/actnet/cvof/";
dstimgDirV = "/mnt/sun-gamma/actnet/cvof/";
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
    
    subset = 'validation'
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
        if ecount>-1:
            srcimgDirV = srcimgDir+'v_'+videoId+'/'
            dstimgDirV = dstimgDir+'v_'+videoId+'/'
            videoInfo = database[videoId]
            if not videoInfo['isnull'] and videoInfo['subset'] == subset:
                vcount+=1
                print srcimgDirV,' ecount ',ecount,videoInfo['subset'],' vcount ',vcount
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
                    src = srcimgDirV+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                    if os.path.isfile(src):
                        #dst = dstimgDirV+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                        imagelists[label].append(src)
    
    filename = baseDir+'lists/{}-motionImageLists.pkl'.format(subset)
    print 'save imglist in ',filename
    with open(filename,'wb') as f:
        pickle.dump(imagelists,f)
        
        
        
def laters():
    max1 = 3; max2 = 5;
    srcImagelists = [[] for i in range(201)];
    dstImagelists = [[] for i in range(201)];
    for action in range(201):
        imagelist = imagelists[action]
        numImages = len(imagelist)
        if action == 200:
            maxallowed = max2
        else:
            maxallowed = max1
        
        if numImages>maxallowed:
            print ' '
            index = np.asarray(np.arange(numImages),dtype='uint32')
            np.random.shuffle(index)
            for i in range(maxallowed):
                src = imagelist[index[i]];
                srcImagelists[action].append(src)
                dst = dstimgDirV+src.split('/')[-1]
                srcImagelists[action].append(src)
        else:
            print ' ok'
            writeImgNamestofile(fid,actionImagelist)
        
    
                    
                    
def genimglist(fid,subset,max1,max2):
    np.random.seed(42)
    maxallowed = max1;
    filename = baseDir+'lists/{}-motionImageLists.pkl'.format(subset)
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
            writeImgNamestofilenMove(fid,actionImagelist,index[:maxallowed])
        else:
            print ' ok'
	    if subset=='training' and len(actionImagelist)<=32000:
	   	 ind = np.random.shuffle([1,2,3,4,5,6])
	    if subset=='validation' and len(actionImagelist)<=16000:
                 ind = np.random.shuffle([1,2,3,4,5,6])
            writeImgNamestofilenMove(fid,actionImagelist)

def writeImgNamestofilenMove(fid,imglist,index=[]):
    if len(index)==0:
        for imgname in imglist:
            src = imgname;
            splits = src.split('/');
            dst = dstimgDirV+splits[-2]+'0iMg0'+splits[-1]
            if not os.path.isfile(dst):
                print 'move',src,dst
                shutil.copy(src,dst)
            fid.write(dst+'\n')
    else:
        for ind in sorted(index):
            imgname = imglist[ind]
            src = imgname;
            splits = src.split('/');
            dst = dstimgDirV+splits[-2]+'0iMg0'+splits[-1]
            if not os.path.isfile(dst):
                print src,dst
                shutil.copy(src,dst)
            fid.write(dst+'\n')
            
def writeImgNamestofile(fid,imglist,index=[]):
    if len(index)==0:
        for imgname in imglist:
            
            fid.write(imgname+'\n')
    else:
        for ind in index:
            imgname = imglist[ind]
            fid.write(imgname+'\n')
def genJointList():
    trainlist = baseDir+'lists/{}-{}-motion.list'.format('train','valid')
    fid = open(trainlist,'wb');
    genimglist(fid,'training',50000,70000)
    genimglist(fid,'validation',25000,35000)
    fid.close()
    
def genlists():
    subset = 'training'
    subset = 'validation'
    trainlist = baseDir+'lists/{}-motion.list'.format(subset)
    fid = open(trainlist,'wb')
    genimglist(fid,subset,10000,45000)

def vgenlists():
    subset = 'training'
    subset = 'validation'
    trainlist = baseDir+'lists/{}-motion.list'.format(subset)
    fid = open(trainlist,'wb')
    genimglist(fid,subset,10000,30000)

def tempgenlists():
    subset = 'training'
    subset = 'validation'
    trainlist = baseDir+'lists/{}-tempmotion.list'.format(subset)
    fid = open(trainlist,'wb')
    genimglist(fid,subset,10000,32000)
    
def genertalabels():
    labellist = baseDir+'lists/labels.list'
    fid = open(labellist,'wb');
    for label in range(201):
        fid.write('ActId'+str(label).zfill(3)+'\n')
    
if __name__=="__main__":
    #writelabels()
    # genimglist()
    tempgenlists()
    # genJointList()
    # genertalabels()
    # checkmogrify()
    # mymogrify()
