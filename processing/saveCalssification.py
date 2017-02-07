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
    predictionfile = '{}data/predictions/{}-{}-{}/{}.list'.format(baseDir,subset,imgtype,str(weight).zfill(5),vidname)
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
    
def readpkl(filename):
    with open(filename) as f:
        data = pickle.load(f)
    return data

def getVideoData(database,indexs,gtlabels,subset,featType):
    if featType == 'MBH':
        filename = baseDir+'data/MBH_Videos_features.hdf5';
        x = np.zeros((20000,65536))
    else:
        filename = baseDir+'data/ImageNetShuffle2016_features.hdf5';
        x = np.zeros((20000,1024))
        
    file = h5py.File(filename,'r')
    features = file['features']
    #print np.shape(features)
    
    count = 0;
    y = dict()
    #features = power_normalize(features)
    for videoId in database.keys():
            videoInfo = database[videoId]
        # if videoInfo['subset'] == subset:
            vkey = 'v_'+videoId;
            ind = indexs[vkey]
            label = gtlabels[videoId]
            #feat = features[ind,:]
            x[count,:] = features[ind,:];
            y[videoId] = count;
            count+=1
            
            
    file.close()
    return x[:count],y

def getScores(database,indexs,gtlabels,subset,featType):
    
    # if 'subset' == 'testing':
    # svmFile = '{}data/train-valSVM-{}.pkl'.format(baseDir,featType)
    # else:
    
    svmFile = '{}data/trainingSVM-{}.pkl'.format(baseDir,featType)
    with open(svmFile,'r') as f:
            clf = pickle.load(f)
    X,videoIndexs = getVideoData(database,indexs,gtlabels,subset,featType)
    print 'shape of the data is is ',np.shape(X)
    #bestlabels = clf.predict(X)
    scores = clf.decision_function(X)
    results = dict()
    results['scores'] = scores
    results['vIndexs'] = videoIndexs
    return results

def getC3dScores(database,indexs,gtlabels,subset,featType):
    
    # if 'subset' == 'testing':
    #     svmFile = '{}data/train-valSVM-{}.pkl'.format(baseDir,featType)
    # else:
    #     svmFile = '{}data/trainingSVM-{}.pkl'.format(baseDir,featType)
    
    
    svmFile = '{}data/trainingSVM-{}.pkl'.format(baseDir,featType)
    with open(svmFile,'r') as f:
            clf = pickle.load(f)
    filename = baseDir+'data/sub_activitynet_v1-3.c3d.hdf5';
    file = h5py.File(filename,'r')
    
    savename = '{}data/ALLpredictions-SVM-{}.hdf5'.format(baseDir,featType)
    sf = h5py.File(savename,'w')
    
    results = dict()
    vcount = 0;
    for videoId in database.keys():
            videoInfo = database[videoId]
            vdict = dict()
        # if not videoInfo['subset'] == 'all':
            vcount+=1
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
            labels = framelabels[indexs]
            # scores = clf.predict_proba(videofeatures)
            scores = clf.decision_function(videofeatures)
            print 'Done ',vkey,vcount
            gp = sf.create_group(videoId)
            gp['scores'] = scores
            gp['indexs'] = indexs
            gp['labels'] = labels
            
    sf.close()


def getExtScores(database,indexs,gtlabels,subset,featType):
    
    weight = 30000;
    imgtype = 'rgb';
    savename = '{}data/predictions-{}-{}.hdf5'.format(baseDir,subset,featType)
    sf = h5py.File(savename,'w')
    
    results = dict()
    vcount = 0;
    for videoId in database.keys():
        videoInfo = database[videoId]
        vdict = dict()
        if videoInfo['subset'] == subset:
            vcount+=1
            vidname = 'v_'+videoId
            print 'doing ',vidname,vcount
            gtlabels,scores = getpredications(subset,imgtype,weight,vidname)
            gp = sf.create_group(videoId)
            gp['scores'] = scores
            gp['labels'] = gtlabels
    sf.close()
    
    
def savePredictions():
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    # 
    subset = 'validation'
    featType = 'MBH'
    predictions = getScores(database,indexs,gtlabels,subset,featType)
    savename = '{}data/ALLpredictions-{}.pkl'.format(baseDir,featType)
    with open(savename,'w') as f:
        pickle.dump(predictions,f)
    
    #########################################
    featType = 'IMS'
    predictions = getScores(database,indexs,gtlabels,subset,featType)
    savename = '{}data/ALLpredictions-{}.pkl'.format(baseDir,featType)
    with open(savename,'w') as f:
        pickle.dump(predictions,f)
    
    ######################################
    
    # featType = 'C3D'
    # getC3dScores(database,indexs,gtlabels,subset,featType)
    
    # savename = '{}data/ALLpredictions-{}-{}.pkl'.format(baseDir,subset,featType)
    # featType = 'EXT'
    # getExtScores(database,indexs,gtlabels,subset,featType)
    savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
    
if __name__=="__main__":
    savePredictions()
    
