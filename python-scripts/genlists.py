'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read annotation from json and
resave annotation with frame level information along with duration info
for eg: a segement described with start frame and end frame
also label it with numeric id rather than text label

'''

import numpy as np
import pickle
import os
import time

baseDir = "/mnt/sun-alpha/actnet/";
imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
# imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
    

    
def genvideolists():
    subset = 'testing'
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    ind = np.arange(0,201)     
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    actionframecount = np.zeros(201,dtype='uint32')
    listname = baseDir+'lists/videolist-'+subset+'.list'
    fid = open(listname,'wb');
    for videoId in database.keys():
            videoInfo = database[videoId]
            if not videoInfo['isnull'] and videoInfo['subset'] == subset:
                vidname = 'v_'+videoId+'\n'
                fid.write(vidname)
    fid.close()
    
if __name__=="__main__":
    genvideolists()
