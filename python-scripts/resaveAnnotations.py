'''
Autor: Gurkirt Singh
Start data: 2nd May 2016
purpose: of this file is read annotation from json and
resave annotation with frame level information along with duration info
for eg: a segement described with start frame and end frame
also label it with numeric id rather than text label

'''

import numpy as np
import cv2 as cv2
import scipy as sp
import scipy.io as sio
import pickle
import os
import json
import shutil

vidDir = "/mnt/sun-alpha/actnet/videos/";
imgDir = "/mnt/earth-beta/Datasets/actnet/images/";
annotFile = "../Evaluation/data/activity_net.v1-3.min.json"


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

def readAnnotFile():
    with open(annotFile) as f:
        annoData = json.load(f)
    taxonomy = annoData["taxonomy"]
    version = annoData["version"]
    database = annoData["database"]
    return taxonomy,version,database
    
def getNumFrames(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened(): 
        print "could not open :",filename
        return -1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return numf

def checkNamelen():
    vidlist = os.listdir(vidDir)
    vidlist = [vid for vid in vidlist if vid.endswith(".mp4")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    count = 0;
    for videname in vidlist:
        basename = videname.split('.mp')[0]
        if len(basename)==13:
            print 'we have culprit in name of ', basename
            count+=1
    print 'culprit count ',count,len(vidlist)
def getTaxonomyDictionary(taxonomy):
    mytaxonomy = dict();
    for entry in taxonomy:
        nodeName = entry['nodeName'];
        mytaxonomy[nodeName] = entry;
    return mytaxonomy

def getNodeNum(taxonomy,actionName):
    
    actionInfo = taxonomy[actionName]
    actionID = actionInfo['nodeId']
    return actionID

def getClassIds():
    taxonomy,version,database = readAnnotFile();
    mytaxonomy = getTaxonomyDictionary(taxonomy);    
    actionIDs = dict();
    for videoID in database.keys():
        videoname = vidDir+'v_'+videoID+'.mp4'
        vidinfo = database[videoID]
        for vidfield in vidinfo.keys():
            if vidfield == 'annotations':
                for annot in vidinfo[vidfield]:
                    label = annot['label']
                    if label in actionIDs.keys():
                        actionIDs[label]['count'] +=1
                        if not actionIDs[label]['nodeId']  == getNodeNum(mytaxonomy,label):
                            RuntimeError('some locha here')
                    else:
                        actionIDs[label] = {'count':1,'nodeId':getNodeNum(mytaxonomy,label)}
    
    classids = dict()
    classnum = 1;
    
    for label in actionIDs.keys():
        actionIDs[label]['class'] = classnum
        print label, ' and count is ',actionIDs[label]['count'],' nodeid id is ',actionIDs[label]['nodeId']
        classnum+=1    
    return actionIDs
    
def main():
    taxonomy,version,database = readAnnotFile()
    mytaxonomy = getTaxonomyDictionary(taxonomy);
    actionIDs = getClassIds() 
    ecount = 0;
    newdatabase = dict();
    verbose = 0
    for videoID in database.keys():
            ecount+=1
        # if ecount<2:
            videoname = vidDir+'v_'+videoID+'.mp4'
            print 'doing ',videoname,' ecount ',ecount
            vidinfo = database[videoID]
            if os.path.isfile(videoname):
                mydict = {'isnull':0}
                numf,width,height,fps = getVidedInfo(videoname)
                if verbose:
                    print numf,width,height,fps
                storageDir = imgDir+'v_'+videoID+"/"
                imgname = storageDir+str(0).zfill(5)+".jpg"
                image = cv2.imread(imgname)
                height,width,depth = np.shape(image)
                newres = [height,width];
                mydict['newResolution'] = newres;
                mydict['numf'] = numf;
                mydict['fps'] = fps;
                myannot = [];                
                for vidfield in vidinfo.keys():
                    if vidfield == 'annotations':
                        for annot in vidinfo[vidfield]:
                            tempsegment = dict()
                            tempsegment['segment'] = annot['segment']
                            tempsegment['label'] = annot['label']
                            segment = annot['segment'];
                            tempsegment['sf'] = max(int(segment[0]*fps),0)
                            tempsegment['ef'] = min(int(segment[1]*fps),numf)
                            tempsegment['nodeid'] = actionIDs[annot['label']]['nodeId']
                            tempsegment['class'] = actionIDs[annot['label']]['class']
                            myannot.append(tempsegment)
                    else:
                        mydict[vidfield] = vidinfo[vidfield]
                mydict['annotations'] = myannot
            else:
                mydict = vidinfo;
                mydict['isnull'] = 1
            
            newdatabase[videoID] = mydict

        
    actNetDB = {'actionIDs':actionIDs,'version':version,'taxonomy':mytaxonomy,'database':newdatabase}
    
    with open('actNet200-V1-3.pkl','wb') as f:
        pickle.dump(actNetDB,f)

    # with open('actNet200-V1-3.pkl','rb') as f:
    #     actNetDB = pickle.load(f)
    
if __name__=="__main__":
    # checkNamelen()
    main()
    # getClassIds()