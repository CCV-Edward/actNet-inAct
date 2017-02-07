'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read frame level predictions and process them to produce a label per video

'''
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os,h5py
import time,json
#import pylab as plt

#######baseDir = "/mnt/sun-alpha/actnet/";
baseDir = "/data/shared/solar-machines/actnet/";
########imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
######## imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
    
def readannos():
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    return actionIDs,taxonomy,database

def getnames():
    fname = baseDir+'data/lists/gtnames.list'
    with open(fname,'rb') as f:
        lines = f.readlines()
    names = []
    for name in lines:
        name = name.rstrip('\n')
        names.append(name)
    # print names
    return names

def getpredications(subset,imgtype,weight,vidname):
    predictionfile = '{}predictions/{}-{}-{}/{}.list'.format(baseDir,subset,imgtype,str(weight).zfill(5),vidname)
    with open(predictionfile) as f:
        lines = f.readlines()
    preds = np.zeros((201,len(lines)),dtype = 'float32')
    labels = np.zeros(len(lines))
    lcount = 0;
    for line in lines:
        splitedline = line.split(' ');
        labels[lcount] = int(splitedline[0])
        wcount = 0;
        # print 'line length ', len(splitedline)
        # print splitedline
        for word in splitedline[1:-1]:
            # print word,
            preds[wcount,lcount] = float(word)
            wcount+=1
        lcount +=1
    return labels,preds

def gettopklabel(preds,k,classtopk):
    scores = np.zeros(200)
    topk = min(classtopk,np.shape(preds)[1]);
    for i in range(200):
        values = preds[i,:];
        values = np.sort(values);
        values = values[::-1]
        scores[i] = np.mean(values[:topk])
    # print scores
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]
    return sortedlabel[:k],sortedscores[:k]

def readpkl(filename):
    with open(filename) as f:
        data = pickle.load(f)
    return data

def getframelabels(videoInfo,numf):
    numf = videoInfo['numf'];
    duration = videoInfo['duration']
    #fps = videoInfo['fps'];
    fps = numf/duration;
    subset = videoInfo['subset']
    framelabels = np.ones(numf,dtype='uint16')*200;
    if subset == 'testing':
        return framelabels
    else:
        annotations = videoInfo['annotations']
        for annot in annotations:
            segment = annot['segment'];
            actionId = annot['class']
            startframe = max(int(segment[0]*fps)-1,0)
            endframe = min(int(segment[1]*fps)+1,numf)
            framelabels[startframe:endframe] = int(actionId)-1
        return framelabels
    
def getC3Ddata(database,indexs,gtlabels,subset):
    
    filename = baseDir+'data/sub_activitynet_v1-3.c3d.hdf5';
    x = np.zeros((5590000,500))
    y = np.zeros(5590000)    
    
    file = h5py.File(filename,'r')
    count = 0;
    vcount  = 0;
    maxnump = 200;
    for videoId in database.keys():
        videoInfo = database[videoId]
        if videoInfo['subset'] == subset:
            # print 'Doing ',vcount
            vcount += 1
            vkey = 'v_'+videoId;
            videofeatures = file[vkey]
            videofeatures = videofeatures['c3d_features']
            numfeat = np.shape(videofeatures)[0]
            numfs = videoInfo['numf']
            framelabels = getframelabels(videoInfo,numfs)
            indexs = np.asarray(np.arange(7,numfs-8,8))
            if abs(len(indexs)-numfeat)>0:
                offset = int(8/(float(numfeat)/len(indexs)))
                #print offset
                indexs = np.asarray(np.linspace(offset,numfs-offset,numfeat),dtype=int)
            #print indexs
            #print len(indexs),len(framelabels),numfeat,vcount,count
            labelIndexs = framelabels[indexs]
            indexs = np.arange(0,numfeat)
            posindexs = indexs[labelIndexs<200]
            np.random.shuffle(posindexs)
            negindexs = indexs[labelIndexs==200]
            np.random.shuffle(negindexs)
            for i in range(min(maxnump,len(posindexs))):
                ind = posindexs[i]
                x[count,:] = videofeatures[ind,:]
                y[count] = 1#labelIndexs[ind]
                count+=1
                ##print labelIndexs[ind]
                
            for i in range(min(maxnump,len(negindexs))):
                ind = negindexs[i]
                x[count,:] = videofeatures[ind,:]
                y[count] = 0#labelIndexs[ind]
                count+=1
                
            
    file.close()
    return x[:count],y[:count]

def processC3D():

    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    print 'getting training data.... '
    xtrain,ytrain = getC3Ddata(database,indexs,gtlabels,'training')
    print 'got it!! and shape is ',np.shape(xtrain)
    print 'getting validation data.... '
    xval,yval = getC3Ddata(database,indexs,gtlabels,'validation')
    print 'got it!! and shape is ',np.shape(xval)
    
    numSamples = np.shape(xval)[0]

    clf = RandomForestClassifier(n_estimators=256,n_jobs=2)
    clf = clf.fit(xtrain, ytrain)
    preds = clf.predict(xval)
    correctPreds = preds == yval;
    print 'Overall Accuracy is ',100*float(np.sum(correctPreds))/numSamples, '% ', ' with RF'
    
    saveName = '{}data/BWtrainingRF-{}.pkl'.format(baseDir,'C3D')
    with open(saveName,'w') as f:
            pickle.dump(clf,f)
            
    print 'training svms '
    Cs = [0.01,0.1,1,10];
    bestclf = {};
    bestscore = 0;
    for cc in Cs:
        clf = LinearSVC(C = cc)
        clf = clf.fit(xtrain, ytrain)
        preds = clf.predict(xval)
        correctPreds = preds == yval;
        score = 100*float(np.sum(correctPreds))/numSamples;
        print 'Overall Accuracy is ',score, '% ', ' C = ',str(cc)
        if score>bestscore:
            bestclf = clf
            bestscore = score
            
    saveName = '{}data/BWtrainingSVM-{}.pkl'.format(baseDir,'C3D')
    with open(saveName,'w') as f:
            pickle.dump(bestclf,f)

if __name__=="__main__":
    #processPredictions()
    processC3D()
    
